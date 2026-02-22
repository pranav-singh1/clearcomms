import { useCallback, useEffect, useRef, useState } from "react";

const ACCEPT = ".wav,.flac,.ogg,.mp3,.m4a,audio/*";
const MIN_RECORD_SAMPLES = 8000;

type BrowserWindow = Window & {
  webkitAudioContext?: typeof AudioContext;
};

type Props = {
  file: File | null;
  onFileChange: (f: File | null) => void;
  onTranscribe: () => void;
  loading: boolean;
  disabled: boolean;
  onMicActiveChange?: (active: boolean) => void;
};

function fileIsAccepted(file: File): boolean {
  return ACCEPT.split(",").some((token) => {
    const value = token.trim().toLowerCase();
    if (value === "audio/*") return file.type.startsWith("audio/");
    return value.startsWith(".") ? file.name.toLowerCase().endsWith(value) : false;
  });
}

function mergeChunks(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function writeAscii(view: DataView, offset: number, text: string) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60).toString().padStart(2, "0");
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

export function UploadZone({ file, onFileChange, onTranscribe, loading, disabled, onMicActiveChange }: Props) {
  const [recording, setRecording] = useState(false);
  const [recordSecs, setRecordSecs] = useState(0);
  const [recordError, setRecordError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sinkRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef<number>(16000);
  const timerRef = useRef<number | null>(null);

  const teardownRecorder = useCallback(async () => {
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (sinkRef.current) {
      sinkRef.current.disconnect();
      sinkRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (contextRef.current) {
      try {
        await contextRef.current.close();
      } catch {
        // noop
      }
      contextRef.current = null;
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && fileIsAccepted(f)) {
        setRecordError(null);
        onMicActiveChange?.(false);
        onFileChange(f);
      } else if (f) {
        setRecordError("Unsupported file format.");
      }
    },
    [onFileChange, onMicActiveChange]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => e.preventDefault(), []);

  const handleInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) {
        onMicActiveChange?.(false);
        onFileChange(null);
      } else if (fileIsAccepted(f)) {
        setRecordError(null);
        onMicActiveChange?.(false);
        onFileChange(f);
      } else {
        setRecordError("Unsupported file format.");
      }
      e.target.value = "";
    },
    [onFileChange, onMicActiveChange]
  );

  const startRecording = useCallback(async () => {
    if (recording) return;
    setRecordError(null);
    chunksRef.current = [];
    setRecordSecs(0);

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support microphone capture.");
      }

      const AudioContextCtor = window.AudioContext || (window as BrowserWindow).webkitAudioContext;
      if (!AudioContextCtor) {
        throw new Error("This browser does not support the Web Audio API.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new AudioContextCtor();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const sink = context.createGain();
      sink.gain.value = 0;

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(context.destination);

      streamRef.current = stream;
      contextRef.current = context;
      sourceRef.current = source;
      processorRef.current = processor;
      sinkRef.current = sink;
      sampleRateRef.current = context.sampleRate;
      timerRef.current = window.setInterval(() => setRecordSecs((value) => value + 1), 1000);
      setRecording(true);
      onMicActiveChange?.(true);
    } catch (error) {
      await teardownRecorder();
      setRecording(false);
      setRecordError(error instanceof Error ? error.message : "Unable to access microphone.");
      onMicActiveChange?.(false);
    }
  }, [onMicActiveChange, recording, teardownRecorder]);

  const stopRecording = useCallback(async () => {
    if (!recording) return;

    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setRecording(false);
    await teardownRecorder();
    const merged = mergeChunks(chunksRef.current);
    chunksRef.current = [];

    if (merged.length < MIN_RECORD_SAMPLES) {
      setRecordError("Recording was too short. Please record again.");
      onMicActiveChange?.(false);
      return;
    }

    const wavBlob = encodeWav(merged, sampleRateRef.current);
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const recordedFile = new File([wavBlob], `mic-${timestamp}.wav`, { type: "audio/wav" });
    setRecordError(null);
    onFileChange(recordedFile);
    onMicActiveChange?.(true);
  }, [onFileChange, onMicActiveChange, recording, teardownRecorder]);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }
      void teardownRecorder();
    };
  }, [teardownRecorder]);


  return (
    <div className="flex flex-col gap-4">
      <div
        className={`border border-dashed transition-colors p-6 flex items-center justify-center text-center font-mono text-sm cursor-pointer ${file ? 'border-defense-accent bg-defense-accent/5 text-white' : 'border-defense-border bg-defense-900 text-defense-muted hover:border-defense-muted'}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept={ACCEPT}
          onChange={handleInput}
          className="hidden"
          id="file-input"
        />
        <label htmlFor="file-input" className="cursor-pointer w-full h-full min-h-[100px] flex items-center justify-center">
          {file ? file.name : "DROP FILE / BROWSE"}
        </label>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <button
          type="button"
          onClick={recording ? () => void stopRecording() : () => void startRecording()}
          disabled={loading}
          className={`py-3 font-semibold text-sm transition-colors font-mono uppercase ${
            recording
              ? "bg-red-800 text-white hover:bg-red-700"
              : "bg-defense-700 text-white hover:bg-defense-600"
          } disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {recording ? `STOP MIC (${formatDuration(recordSecs)})` : "RECORD MIC"}
        </button>

        <button
          type="button"
          onClick={onTranscribe}
          disabled={disabled || loading || recording}
          className="py-3 bg-white text-black font-semibold text-sm hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-mono uppercase"
        >
          {loading ? "INITIALIZING..." : recording ? "STOP MIC FIRST" : "EXECUTE"}
        </button>
      </div>

      <p className="text-xs text-defense-muted font-mono">
        Recordings are saved as WAV and appear in the existing Original Input player.
      </p>

      {recordError && (
        <div className="p-3 bg-red-950/30 border border-red-900/50 text-red-500 text-xs font-mono">
          MIC: {recordError}
        </div>
      )}
    </div>
  );
}
