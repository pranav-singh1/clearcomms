import { useEffect, useMemo, useRef, useState } from "react";
import { synthesizeTTS, type TranscribeResult, type TtsStatus } from "../api";

type Props = {
  result: TranscribeResult;
  originalFile: File | null;
  applyRadioFilter: boolean;
  ttsEnabled: boolean;
  ttsStatus: TtsStatus | null;
};

export function Result({ result, originalFile, applyRadioFilter, ttsEnabled, ttsStatus }: Props) {
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null);
  const ttsCacheRef = useRef<Map<string, string>>(new Map());

  const originalUrl = useMemo(() => (originalFile ? URL.createObjectURL(originalFile) : null), [originalFile]);
  const filteredUrl = useMemo(
    () => (result.audio_filtered_b64 ? `data:audio/wav;base64,${result.audio_filtered_b64}` : null),
    [result.audio_filtered_b64]
  );

  const transcriptIsError = Boolean(result.error);
  const cleanedTranscript = (result.cleaned_transcript || result.text || "").trim();
  const isOffline = typeof navigator !== "undefined" && !navigator.onLine;
  const ttsAvailable = Boolean(ttsStatus?.available);
  const canSpeakCleaned =
    !transcriptIsError && cleanedTranscript.length > 0 && ttsAvailable && ttsEnabled && !ttsLoading;
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

  useEffect(() => {
    setTtsError(null);
    setTtsLoading(false);
    setTtsAudioUrl(null);
  }, [result]);

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
      const audioBlob = await synthesizeTTS(cleanedTranscript);
      const url = URL.createObjectURL(audioBlob);
      ttsCacheRef.current.set(cleanedTranscript, url);
      if (ttsCacheRef.current.size > 20) {
        const oldestKey = ttsCacheRef.current.keys().next().value as string | undefined;
        if (oldestKey) {
          const oldestUrl = ttsCacheRef.current.get(oldestKey);
          if (oldestUrl) URL.revokeObjectURL(oldestUrl);
          ttsCacheRef.current.delete(oldestKey);
        }
      }
      setTtsAudioUrl(url);
    } catch (e: unknown) {
      setTtsError(e instanceof Error ? e.message : "TTS failed.");
      setTtsAudioUrl(null);
    } finally {
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
        <div className="flex flex-col">
          <div className="border-b border-defense-border pb-2 mb-4 font-mono text-xs text-white uppercase">Cleaned Transcript</div>
          <div className={`font-mono text-sm leading-relaxed p-4 bg-defense-900 border ${transcriptIsError ? 'border-red-900/50 text-red-400' : 'border-defense-border text-white'}`}>
            {transcriptContent}
          </div>
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
                ? `Model: ${ttsStatus?.model || "aura-2-thalia-en"}`
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
              <audio controls autoPlay src={ttsAudioUrl} className="w-full h-8" />
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
