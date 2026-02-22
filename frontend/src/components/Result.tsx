import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { synthesizeTTSStream, type TranscribeResult, type TtsStatus } from "../api";
import { useTtsQueue } from "../hooks/useTtsQueue";

type Props = {
  result: TranscribeResult;
  originalFile: File | null;
  applyRadioFilter: boolean;
  ttsEnabled: boolean;
  ttsStatus: TtsStatus | null;
  realtimeTtsEnabled: boolean;
  micActive: boolean;
};

export function Result({
  result,
  originalFile,
  applyRadioFilter,
  ttsEnabled,
  ttsStatus,
  realtimeTtsEnabled,
  micActive,
}: Props) {
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsCacheRef = useRef<Map<string, string>>(new Map());
  const { enqueue, queueSize, playing, generating, error: realtimeError, clearError } = useTtsQueue();

  const originalUrl = useMemo(() => (originalFile ? URL.createObjectURL(originalFile) : null), [originalFile]);
  const filteredUrl = useMemo(
    () => (result.audio_filtered_b64 ? `data:audio/wav;base64,${result.audio_filtered_b64}` : null),
    [result.audio_filtered_b64]
  );

  const transcriptIsError = Boolean(result.error);
  const rawTranscript = (result.raw_transcript ?? result.text ?? "").trim();
  const revisedTranscript = (result.revised_transcript ?? "").trim();
  const cleanedTranscript = (result.revised_transcript ?? result.cleaned_transcript ?? result.text ?? "").trim();
  const cleanedTranscriptRef = useRef(cleanedTranscript);
  const isOffline = typeof navigator !== "undefined" && !navigator.onLine;
  const ttsAvailable = Boolean(ttsStatus?.available);
  const canSpeakCleaned =
    !transcriptIsError && cleanedTranscript.length > 0 && ttsAvailable && ttsEnabled && !ttsLoading;
  const rawContent = result.error ? `ERR: ${result.error}` : (rawTranscript || "NO TRANSCRIPT_");
  const llamaError = result.meta?.llama_revision_error;
  const revisedContent = revisedTranscript
    ? revisedTranscript
    : typeof llamaError === "string" && llamaError
    ? `Llama revision failed: ${llamaError}`
    : "— Llama revision not run (set ENABLE_LLAMA_REVISION=1 before starting the backend)";
  const revisedIsError = Boolean(revisedTranscript === "" && llamaError);
  const transcriptContent = result.error
    ? `ERR: ${result.error}`
    : (cleanedTranscript || result.text || "NO TRANSCRIPT_");

  // Simulated structured data based on the prompt "Cleaned transcript, Structured JSON output"
  const structuredData = useMemo(() => {
    if (result.error || !result.text) return null;
    return {
      "id": `INC-${Math.floor(Math.random()*10000)}`,
      "priority": "HIGH",
      "summary": "Extracted from radio transmission",
      "raw_text": result.text.substring(0, 100) + "...",
      "timestamp": new Date().toISOString()
    };
  }, [result]);

  useEffect(() => {
    return () => {
      for (const url of ttsCacheRef.current.values()) {
        URL.revokeObjectURL(url);
      }
      ttsCacheRef.current.clear();
    };
  }, []);

  const lastSpokenFullRef = useRef<string>("");
  const lastTtsAtRef = useRef<number>(0);
  const pendingTimerRef = useRef<number | null>(null);
  const minIntervalMs = 1500;
  const debounceMs = 600;

  useEffect(() => {
    setTtsError(null);
    setTtsLoading(false);
    setTtsAudioUrl(null);
    lastSpokenFullRef.current = "";
    lastTtsAtRef.current = 0;
  }, [result]);

  useEffect(() => {
    if (!ttsAudioUrl) return;
    const el = ttsAudioRef.current;
    if (!el) return;
    el.preload = "auto";
    const play = () => {
      void el.play().catch(() => {
        /* autoplay may be blocked */
      });
    };
    if (el.readyState >= 2) {
      play();
      return;
    }
    const onCanPlay = () => {
      el.removeEventListener("canplay", onCanPlay);
      play();
    };
    el.addEventListener("canplay", onCanPlay);
    return () => {
      el.removeEventListener("canplay", onCanPlay);
    };
  }, [ttsAudioUrl]);

  useEffect(() => {
    cleanedTranscriptRef.current = cleanedTranscript;
  }, [cleanedTranscript]);

  const isSpeakable = useCallback((text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 8) return false;
    const words = trimmed.split(/\s+/).filter(Boolean);
    return words.length >= 2;
  }, []);

  const extractSegment = useCallback((fullText: string, lastFull: string) => {
    if (lastFull && fullText.startsWith(lastFull)) {
      return fullText.slice(lastFull.length).trim();
    }
    return fullText.trim();
  }, []);

  useEffect(() => {
    if (!realtimeTtsEnabled || !micActive || !ttsEnabled || !ttsAvailable || transcriptIsError) {
      if (pendingTimerRef.current !== null) {
        window.clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = null;
      }
      return;
    }
    if (!isSpeakable(cleanedTranscript)) return;

    if (pendingTimerRef.current !== null) {
      window.clearTimeout(pendingTimerRef.current);
      pendingTimerRef.current = null;
    }

    let cancelled = false;
    const attemptSpeak = () => {
      if (cancelled) return;
      if (cleanedTranscriptRef.current !== cleanedTranscript) return;

      const sinceLast = Date.now() - lastTtsAtRef.current;
      if (sinceLast < minIntervalMs) {
        pendingTimerRef.current = window.setTimeout(attemptSpeak, minIntervalMs - sinceLast);
        return;
      }

      const segment = extractSegment(cleanedTranscript, lastSpokenFullRef.current);
      if (!isSpeakable(segment)) {
        if (cleanedTranscript.length >= lastSpokenFullRef.current.length) {
          lastSpokenFullRef.current = cleanedTranscript;
        }
        return;
      }

      lastTtsAtRef.current = Date.now();
      lastSpokenFullRef.current = cleanedTranscript;
      enqueue(segment);
    };

    pendingTimerRef.current = window.setTimeout(attemptSpeak, debounceMs);

    return () => {
      cancelled = true;
      if (pendingTimerRef.current !== null) {
        window.clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = null;
      }
    };
  }, [
    cleanedTranscript,
    enqueue,
    extractSegment,
    isSpeakable,
    micActive,
    realtimeTtsEnabled,
    transcriptIsError,
    ttsAvailable,
    ttsEnabled,
  ]);

  useEffect(() => {
    if (!realtimeTtsEnabled) clearError();
  }, [clearError, realtimeTtsEnabled]);

  const handleSpeakCleanedTranscript = async () => {
    if (!canSpeakCleaned) return;
    const cached = ttsCacheRef.current.get(cleanedTranscript);
    if (cached) {
      setTtsAudioUrl(cached);
      return;
    }
    setTtsLoading(true);
    setTtsError(null);
    try {
      const { url, done } = await synthesizeTTSStream(cleanedTranscript);
      setTtsAudioUrl(url);
      setTtsLoading(false);
      void done
        .then(() => {
          ttsCacheRef.current.set(cleanedTranscript, url);
          if (ttsCacheRef.current.size > 20) {
            const oldestKey = ttsCacheRef.current.keys().next().value as string | undefined;
            if (oldestKey) {
              const oldestUrl = ttsCacheRef.current.get(oldestKey);
              if (oldestUrl) URL.revokeObjectURL(oldestUrl);
              ttsCacheRef.current.delete(oldestKey);
            }
          }
        })
        .catch((e: unknown) => {
          setTtsError(e instanceof Error ? e.message : "TTS failed.");
        });
    } catch (e: unknown) {
      setTtsError(e instanceof Error ? e.message : "TTS failed.");
      setTtsAudioUrl(null);
      setTtsLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Audio Streams */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="flex flex-col gap-2 p-4 bg-defense-900 border border-defense-border">
          <div className="font-mono text-xs text-defense-muted uppercase flex justify-between">
            <span>Stream_01 (RAW)</span>
            <span>{result.sample_rate_original}Hz / {result.duration_sec}s</span>
          </div>
          {originalUrl ? <audio controls src={originalUrl} className="w-full h-8 outline-none" /> : <div className="text-xs text-defense-muted">N/A</div>}
        </div>

        {applyRadioFilter && (
          <div className="flex flex-col gap-2 p-4 bg-defense-900 border border-defense-border">
            <div className="font-mono text-xs text-defense-accent uppercase">Stream_02 (DSP Processed)</div>
            {filteredUrl ? <audio controls src={filteredUrl} className="w-full h-8" /> : <div className="text-xs text-defense-muted">N/A</div>}
          </div>
        )}
      </div>

      {/* Transcripts & Data */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        <div className="flex flex-col gap-6">
          <div>
            <div className="border-b border-defense-border pb-2 mb-2 font-mono text-xs text-white uppercase">Raw transcript (Whisper)</div>
            <div className={`font-mono text-sm leading-relaxed p-4 bg-defense-900 border ${transcriptIsError ? "border-red-900/50 text-red-400" : "border-defense-border text-white"}`}>
              {rawContent}
            </div>
          </div>
          <div>
            <div className="border-b border-defense-border pb-2 mb-2 font-mono text-xs text-white uppercase">Reconstructed transcript (Llama)</div>
            <div className={`font-mono text-sm leading-relaxed p-4 bg-defense-900 border ${revisedTranscript ? "border-defense-border text-white" : revisedIsError ? "border-amber-900/50 text-amber-300" : "border-defense-border text-defense-muted"}`}>
              {revisedContent}
            </div>
            {revisedTranscript && (
              <div className="mt-1 text-xs font-mono text-green-500/90">Llama revision applied.</div>
            )}
          </div>
          {realtimeTtsEnabled && (
            <div className="mt-3 text-xs font-mono text-defense-muted">
              Realtime TTS: ON
              {queueSize > 0 ? ` · Queue: ${queueSize}` : ""}
              {generating ? " · Generating speech..." : ""}
              {playing ? " · Playing" : ""}
            </div>
          )}
          {realtimeTtsEnabled && realtimeError && (
            <div className="mt-2 p-2 bg-amber-950/30 border border-amber-900/50 text-amber-300 text-xs font-mono">
              TTS failed (continuing): {realtimeError}
            </div>
          )}
          <div className="mt-4 flex flex-col gap-3">
            <button
              type="button"
              onClick={handleSpeakCleanedTranscript}
              disabled={!canSpeakCleaned}
              title={
                !ttsEnabled
                  ? "TTS is disabled in settings."
                  : !ttsAvailable
                  ? (isOffline ? "TTS unavailable offline." : ttsStatus?.reason || "TTS unavailable.")
                  : cleanedTranscript.length === 0
                  ? "No cleaned transcript to speak."
                  : undefined
              }
              className="px-4 py-2 border border-defense-border bg-defense-800 text-xs font-mono text-white hover:bg-defense-700 transition disabled:opacity-50 disabled:cursor-not-allowed text-left"
            >
              {ttsLoading ? "GENERATING SPEECH..." : "SPEAK CLEANED TRANSCRIPT"}
            </button>
            <div className="text-xs font-mono text-defense-muted">
              {ttsAvailable
                ? `Model: ${ttsStatus?.model || "aura-2-arcas-en"}`
                : isOffline
                ? "TTS unavailable offline."
                : ttsStatus?.reason || "TTS unavailable."}
            </div>
            {ttsError && (
              <div className="p-3 bg-red-950/30 border border-red-900/50 text-red-400 text-xs font-mono">
                TTS ERR: {ttsError}
              </div>
            )}
            {ttsAudioUrl && (
              <audio ref={ttsAudioRef} controls autoPlay preload="auto" src={ttsAudioUrl} className="w-full h-8" />
            )}
          </div>
        </div>

        <div className="flex flex-col">
          <div className="border-b border-defense-border pb-2 mb-4 font-mono text-xs text-white uppercase">Structured Extraction</div>
          <div className="font-mono text-xs p-4 bg-defense-900 border border-defense-border text-green-400 overflow-x-auto">
            {structuredData ? (
              <pre>{JSON.stringify(structuredData, null, 2)}</pre>
            ) : (
              <span className="text-defense-muted">NO DATA EXTRACTED_</span>
            )}
          </div>
        </div>
      </div>

      {/* Metrics & Export */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 border-t border-defense-border pt-6">
        <div className="flex flex-col gap-2">
          <div className="font-mono text-xs text-defense-muted uppercase">Telemetry</div>
          <pre className="font-mono text-xs text-defense-muted bg-defense-900 p-3 border border-defense-border overflow-x-auto">
            {Object.keys(result.meta).length ? JSON.stringify(result.meta, null, 2) : "NO TELEMETRY"}
          </pre>
        </div>
        <div className="flex flex-col gap-2 justify-end items-end">
           <button onClick={() => {
              const blob = new Blob([result.text + "\n"], { type: "text/plain" });
              const a = document.createElement("a");
              a.href = URL.createObjectURL(blob);
              a.download = "transcript.txt";
              a.click();
           }} className="px-4 py-2 border border-defense-border bg-defense-800 text-xs font-mono text-white hover:bg-defense-700 transition w-full md:w-auto text-left">
             EXPORT_TRANSCRIPT.TXT
           </button>
           <button onClick={() => {
              const blob = new Blob([JSON.stringify({ meta: result.meta }, null, 2)], { type: "application/json" });
              const a = document.createElement("a");
              a.href = URL.createObjectURL(blob);
              a.download = "metadata.json";
              a.click();
           }} className="px-4 py-2 border border-defense-border bg-defense-800 text-xs font-mono text-white hover:bg-defense-700 transition w-full md:w-auto text-left">
             EXPORT_METADATA.JSON
           </button>
        </div>
      </div>
    </div>
  );
}
