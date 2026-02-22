const API_BASE = "";

export type ModelStatus = { models_found: boolean };

export type TranscribeResult = {
  success: boolean;
  error: string | null;
  text: string;
  cleaned_transcript?: string;
  meta: Record<string, number | string>;
  audio_prepared_b64: string | null;
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

export async function synthesizeOfflineTTS(text: string): Promise<Blob> {
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

export async function uploadAndTranscribe(
  file: File,
  applyRadioFilter: boolean,
  normalize: boolean
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("apply_radio_filter", String(applyRadioFilter));
  form.append("normalize", String(normalize));

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
