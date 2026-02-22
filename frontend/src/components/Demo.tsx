import { useState, useCallback, useEffect } from "react";
import { uploadAndTranscribe, ttsStatus, type TranscribeResult, type TtsStatus } from "../api";
import { Controls } from "./Controls";
import { UploadZone } from "./UploadZone";
import { Result } from "./Result";

export function Demo() {
  const [applyRadioFilter, setApplyRadioFilter] = useState(false);
  const [radioIntensity, setRadioIntensity] = useState(50);
  const [normalize, setNormalize] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscribeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [ttsState, setTtsState] = useState<TtsStatus | null>(null);

  useEffect(() => {
    ttsStatus()
      .then((status) => {
        setTtsState(status);
        if (!status.available) setTtsEnabled(false);
      })
      .catch(() => {
        setTtsState({ available: false, model: "aura-2-thalia-en", reason: "Unavailable" });
        setTtsEnabled(false);
      });
  }, []);

  const handleFileChange = useCallback((f: File | null) => {
    setFile(f);
    setResult(null);
    setError(null);
    // Auto-configure DSP based on source:
    // mic recordings are already clean so run them through the radio filter pipeline;
    // uploaded files (already radio-distorted) skip DSP by default.
    if (f) {
      setApplyRadioFilter(f.name.startsWith("mic-"));
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const source: "mic" | "file" = file.name.startsWith("mic-") ? "mic" : "file";
    try {
      const data = await uploadAndTranscribe(file, applyRadioFilter, normalize, source, radioIntensity);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Transcription failed.");
    } finally {
      setLoading(false);
    }
  }, [file, applyRadioFilter, normalize, radioIntensity]);

  return (
    <section id="demo" className="py-12 border-b border-defense-border/50">
      <div className="mb-10">
        <h2 className="text-sm font-mono text-defense-accent uppercase tracking-widest mb-2">03 / Interactive Console</h2>
        <h3 className="text-3xl font-semibold text-white">System Demo</h3>
      </div>
      
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="lg:w-1/3 flex flex-col gap-6">
          <div className="bg-defense-800 border border-defense-border p-6">
            <h4 className="font-mono text-sm tracking-widest text-white mb-4 uppercase">Configuration</h4>
            <Controls
              applyRadioFilter={applyRadioFilter}
              setApplyRadioFilter={setApplyRadioFilter}
              radioIntensity={radioIntensity}
              setRadioIntensity={setRadioIntensity}
              normalize={normalize}
              setNormalize={setNormalize}
              ttsEnabled={ttsEnabled}
              setTtsEnabled={setTtsEnabled}
              ttsStatus={ttsState}
            />
          </div>
          
          <div className="bg-defense-800 border border-defense-border p-6">
            <h4 className="font-mono text-sm tracking-widest text-white mb-4 uppercase">Input Feed</h4>
            <UploadZone
              file={file}
              onFileChange={handleFileChange}
              onTranscribe={handleSubmit}
              loading={loading}
              disabled={!file}
            />
            {error && (
              <div className="mt-4 p-3 bg-red-950/30 border border-red-900/50 text-red-500 text-sm font-mono">
                ERR: {error}
              </div>
            )}
          </div>
        </div>

        <div className="lg:w-2/3 bg-defense-800 border border-defense-border flex flex-col min-h-[400px]">
          <div className="border-b border-defense-border p-3 flex items-center gap-4">
            <span className="font-mono text-xs text-defense-muted uppercase">Output Buffer</span>
            {loading && <span className="text-defense-accent text-xs font-mono animate-pulse">PROCESSING...</span>}
          </div>
          <div className="p-6 flex-1 flex flex-col">
            {!result && !loading ? (
              <div className="flex-1 flex items-center justify-center text-defense-muted font-mono text-sm border border-dashed border-defense-border">
                AWAITING AUDIO INPUT
              </div>
            ) : (
              result && (
                <Result 
                  result={result} 
                  originalFile={file} 
                  applyRadioFilter={applyRadioFilter}
                  ttsEnabled={ttsEnabled}
                  ttsStatus={ttsState}
                />
              )
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
