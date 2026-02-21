import numpy as np
import os
import queue
import sounddevice as sd
import sys
import threading
import time
import yaml
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from model import make_whisper_app  # <-- use the factory below


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


class LiveTranscriber:
    def __init__(self):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        # audio settings
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.channels = cfg.get("channels", 1)
        self.blocksize = cfg.get("blocksize", 1600)

        # utterance/VAD settings
        self.frame_ms = cfg.get("frame_ms", 20)
        self.vad_rms_threshold = cfg.get("vad_rms_threshold", 0.010)
        self.min_speech_ms = cfg.get("min_speech_ms", 250)
        self.hangover_ms = cfg.get("hangover_ms", 500)
        self.preroll_ms = cfg.get("preroll_ms", 200)
        self.max_utterance_sec = cfg.get("max_utterance_sec", 12)

        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000.0)
        self.min_speech_frames = max(1, int(self.min_speech_ms / self.frame_ms))
        self.hangover_frames = max(1, int(self.hangover_ms / self.frame_ms))
        self.preroll_frames = max(1, int(self.preroll_ms / self.frame_ms))
        self.max_utt_frames = int(self.max_utterance_sec * 1000 / self.frame_ms)

        # threading
        self.max_workers = cfg.get("max_workers", 1)
        self.queue_timeout = cfg.get("queue_timeout", 0.5)
        self.max_queue_chunks = cfg.get("max_queue_chunks", 50)

        # model paths
        self.encoder_path = cfg.get("encoder_path", "models/WhisperEncoder.onnx")
        self.decoder_path = cfg.get("decoder_path", "models/WhisperDecoder.onnx")
        self.model_variant = cfg.get("model_variant", "base_en")

        if not os.path.exists(self.encoder_path):
            sys.exit(f"Encoder model not found at {self.encoder_path}.")
        if not os.path.exists(self.decoder_path):
            sys.exit(f"Decoder model not found at {self.decoder_path}.")

        print("Loading Whisper (QNN EP)...")
        self.whisper = make_whisper_app(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            variant=self.model_variant,
            cfg=cfg,
        )

        # queue + stop
        self.audio_queue = queue.Queue(maxsize=self.max_queue_chunks)
        self.stop_event = threading.Event()

        # printing order
        self._seq = 0
        self._next_to_print = 0
        self._results = {}
        self._lock = threading.Lock()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        if self.stop_event.is_set():
            return
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # drop oldest audio to keep latency bounded
            try:
                _ = self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(indata.copy())
            except Exception:
                pass

    def transcribe_utterance(self, seq: int, audio: np.ndarray):
        t0 = time.time()
        transcript = self.whisper.transcribe(audio, self.sample_rate)
        ms = (time.time() - t0) * 1000.0
        return seq, transcript.strip(), ms, len(audio) / self.sample_rate

    def printer_loop(self):
        # print transcripts in order even if futures return out of order
        while not self.stop_event.is_set():
            with self._lock:
                while self._next_to_print in self._results:
                    transcript, ms, dur = self._results.pop(self._next_to_print)
                    self._next_to_print += 1
                    if transcript:
                        print(f"[{dur:.2f}s | {ms:.0f}ms] {transcript}")
            time.sleep(0.02)

    def process_loop(self):
        # frame buffers
        buf = np.empty((0,), dtype=np.float32)
        preroll = deque(maxlen=self.preroll_frames)
        in_speech = False
        speech_frames = []
        speech_run = 0
        silence_run = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = []

            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=self.queue_timeout).flatten().astype(np.float32)
                    buf = np.concatenate([buf, chunk])

                    # process into 20ms frames
                    while len(buf) >= self.frame_samples:
                        frame = buf[:self.frame_samples]
                        buf = buf[self.frame_samples:]

                        frame_rms = rms(frame)
                        is_speech = frame_rms >= self.vad_rms_threshold

                        preroll.append(frame)

                        if not in_speech:
                            if is_speech:
                                speech_run += 1
                                if speech_run >= self.min_speech_frames:
                                    in_speech = True
                                    silence_run = 0
                                    # start utterance with preroll
                                    speech_frames = list(preroll)
                            else:
                                speech_run = 0
                        else:
                            # we are in utterance
                            speech_frames.append(frame)

                            if is_speech:
                                silence_run = 0
                            else:
                                silence_run += 1

                            # end utterance if trailing silence long enough OR utterance too long
                            if silence_run >= self.hangover_frames or len(speech_frames) >= self.max_utt_frames:
                                utter = np.concatenate(speech_frames, axis=0)

                                seq = self._seq
                                self._seq += 1

                                # submit transcription
                                fut = ex.submit(self.transcribe_utterance, seq, utter)
                                futures.append(fut)

                                # reset state
                                in_speech = False
                                speech_run = 0
                                silence_run = 0
                                speech_frames = []
                                preroll.clear()

                    # harvest completed futures
                    new_futures = []
                    for f in futures:
                        if f.done():
                            seq, text, ms, dur = f.result()
                            with self._lock:
                                self._results[seq] = (text, ms, dur)
                        else:
                            new_futures.append(f)
                    futures = new_futures

                except queue.Empty:
                    continue

            # flush remaining
            for f in futures:
                seq, text, ms, dur = f.result()
                with self._lock:
                    self._results[seq] = (text, ms, dur)

    def run(self):
        printer_thread = threading.Thread(target=self.printer_loop, daemon=True)
        printer_thread.start()

        process_thread = threading.Thread(target=self.process_loop, daemon=True)
        process_thread.start()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self.audio_callback,
            ):
                print("Listening... (Ctrl+C to stop)")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_event.set()
            process_thread.join()
            print("Stopped.")


if __name__ == "__main__":
    LiveTranscriber().run()
