const API_BASE = "";

export type ModelStatus = { models_found: boolean };
export type TtsStatus = { available: boolean; model: string; reason?: string | null };

export type TranscribeResult = {
  success: boolean;
  error: string | null;
  text: string;
  cleaned_transcript?: string;
  raw_transcript?: string | null;
  revised_transcript?: string | null;
  llama_revision_available?: boolean;
  meta: Record<string, number | string>;
  audio_filtered_b64: string | null;
  apply_radio_filter: boolean;
  duration_sec: number;
  sample_rate_original: number;
};

export async function modelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/api/model-status`);
  if (!res.ok) throw new Error("Failed to fetch model status");
  return res.json();
}

export async function ttsStatus(): Promise<TtsStatus> {
  const res = await fetch(`${API_BASE}/api/tts-status`);
  if (!res.ok) throw new Error("Failed to fetch TTS status");
  return res.json();
}

export async function synthesizeTTS(text: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const responseText = await res.text();
    let msg = responseText;
    try {
      const j = JSON.parse(responseText);
      if (j.detail) msg = Array.isArray(j.detail) ? j.detail.map((d: { msg?: string }) => d.msg).join(" ") : j.detail;
    } catch {
      /* use responseText */
    }
    throw new Error(msg || `TTS request failed: ${res.status}`);
  }

  return res.blob();
}

type TtsStreamHandle = {
  url: string;
  done: Promise<void>;
  revoke: () => void;
};

export async function synthesizeTTSStream(text: string): Promise<TtsStreamHandle> {
  const mimeType = "audio/mpeg";
  if (typeof MediaSource === "undefined" || !MediaSource.isTypeSupported(mimeType)) {
    const audioBlob = await synthesizeTTS(text);
    const url = URL.createObjectURL(audioBlob);
    return { url, done: Promise.resolve(), revoke: () => URL.revokeObjectURL(url) };
  }

  const mediaSource = new MediaSource();
  const url = URL.createObjectURL(mediaSource);
  let resolveDone: () => void = () => {};
  let rejectDone: (err: Error) => void = () => {};
  const done = new Promise<void>((resolve, reject) => {
    resolveDone = resolve;
    rejectDone = reject;
  });

  const startStream = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/tts-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const responseText = await res.text();
        let msg = responseText;
        try {
          const j = JSON.parse(responseText);
          if (j.detail)
            msg = Array.isArray(j.detail) ? j.detail.map((d: { msg?: string }) => d.msg).join(" ") : j.detail;
        } catch {
          /* use responseText */
        }
        throw new Error(msg || `TTS request failed: ${res.status}`);
      }
      if (!res.body) {
        throw new Error("Streaming not supported by this browser.");
      }

      const sourceBuffer = mediaSource.addSourceBuffer(mimeType);
      const reader = res.body.getReader();

      const appendChunk = (chunk: Uint8Array) =>
        new Promise<void>((resolve, reject) => {
          const onError = () => {
            sourceBuffer.removeEventListener("error", onError);
            sourceBuffer.removeEventListener("updateend", onUpdateEnd);
            reject(new Error("Failed to append TTS audio."));
          };
          const onUpdateEnd = () => {
            sourceBuffer.removeEventListener("error", onError);
            sourceBuffer.removeEventListener("updateend", onUpdateEnd);
            resolve();
          };
          sourceBuffer.addEventListener("error", onError, { once: true });
          sourceBuffer.addEventListener("updateend", onUpdateEnd, { once: true });
          const buffer = chunk.slice();
          sourceBuffer.appendBuffer(buffer);
        });

      while (true) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;
        if (value && value.length > 0) {
          if (sourceBuffer.updating) {
            await new Promise<void>((resolve) => sourceBuffer.addEventListener("updateend", () => resolve(), { once: true }));
          }
          await appendChunk(value);
        }
      }

      if (sourceBuffer.updating) {
        await new Promise<void>((resolve) => sourceBuffer.addEventListener("updateend", () => resolve(), { once: true }));
      }
      mediaSource.endOfStream();
      resolveDone();
    } catch (err) {
      const error = err instanceof Error ? err : new Error("TTS stream failed.");
      if (mediaSource.readyState === "open") {
        try {
          mediaSource.endOfStream("decode");
        } catch {
          /* ignore */
        }
      }
      rejectDone(error);
    }
  };

  mediaSource.addEventListener(
    "sourceopen",
    () => {
      void startStream();
    },
    { once: true }
  );

  return { url, done, revoke: () => URL.revokeObjectURL(url) };
}

export async function uploadAndTranscribe(
  file: File,
  applyRadioFilter: boolean,
  normalize: boolean,
  source: "mic" | "file" = "file",
  radioIntensity: number = 50
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("apply_radio_filter", String(applyRadioFilter));
  form.append("normalize", String(normalize));
  form.append("source", source);
  form.append("radio_intensity", String(radioIntensity));

  const res = await fetch(`${API_BASE}/api/transcribe`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try {
      const j = JSON.parse(text);
      if (j.detail) msg = Array.isArray(j.detail) ? j.detail.map((d: { msg?: string }) => d.msg).join(" ") : j.detail;
    } catch {
      /* use text */
    }
    throw new Error(msg || `Request failed: ${res.status}`);
  }

  return res.json();
}

export async function reviseTranscript(transcript: string): Promise<{ revised_transcript: string }> {
  const res = await fetch(`${API_BASE}/api/revise`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript }),
  });
  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try {
      const j = JSON.parse(text);
      if (j.detail) msg = Array.isArray(j.detail) ? j.detail.map((d: { msg?: string }) => d.msg).join(" ") : j.detail;
    } catch {
      /* use text */
    }
    throw new Error(msg || `Revision failed: ${res.status}`);
  }
  return res.json();
}
