import { useState, useEffect } from "react";
import { modelStatus, type TtsStatus } from "../api";

type Props = {
  applyRadioFilter: boolean;
  setApplyRadioFilter: (v: boolean) => void;
  normalize: boolean;
  setNormalize: (v: boolean) => void;
  ttsEnabled: boolean;
  setTtsEnabled: (v: boolean) => void;
  ttsStatus: TtsStatus | null;
};

export function Controls({
  applyRadioFilter,
  setApplyRadioFilter,
  normalize,
  setNormalize,
  ttsEnabled,
  setTtsEnabled,
  ttsStatus,
}: Props) {
  const [modelOk, setModelOk] = useState<boolean | null>(null);

  useEffect(() => {
    modelStatus().then(res => setModelOk(res.models_found)).catch(() => setModelOk(false));
  }, []);

  return (
    <div className="flex flex-col gap-4 font-mono text-sm">
      <label className="flex items-start gap-3 cursor-pointer group">
        <input
          type="checkbox"
          checked={applyRadioFilter}
          onChange={(e) => setApplyRadioFilter(e.target.checked)}
          className="mt-1 appearance-none w-4 h-4 border border-defense-border checked:bg-defense-accent checked:border-defense-accent transition-colors relative"
        />
        <div className="flex flex-col gap-1">
          <span className="text-white group-hover:text-defense-accent transition-colors">Apply DSP Bandpass</span>
          <span className="text-xs text-defense-muted">Filters static and applies gate logic</span>
        </div>
      </label>

      <label className="flex items-start gap-3 cursor-pointer group">
        <input
          type="checkbox"
          checked={normalize}
          onChange={(e) => setNormalize(e.target.checked)}
          className="mt-1 appearance-none w-4 h-4 border border-defense-border checked:bg-defense-accent checked:border-defense-accent transition-colors relative"
        />
        <div className="flex flex-col gap-1">
          <span className="text-white group-hover:text-defense-accent transition-colors">Peak Normalization</span>
          <span className="text-xs text-defense-muted">Auto-level input loudness before ASR</span>
        </div>
      </label>

      <label className="flex items-start gap-3 cursor-pointer group">
        <input
          type="checkbox"
          checked={ttsEnabled}
          onChange={(e) => setTtsEnabled(e.target.checked)}
          disabled={!ttsStatus?.available}
          className="mt-1 appearance-none w-4 h-4 border border-defense-border checked:bg-defense-accent checked:border-defense-accent transition-colors relative disabled:opacity-50"
        />
        <div className="flex flex-col gap-1">
          <span className="text-white group-hover:text-defense-accent transition-colors">TTS (online)</span>
          <span className="text-xs text-defense-muted">
            {ttsStatus
              ? ttsStatus.available
                ? `Deepgram ${ttsStatus.model}`
                : ttsStatus.reason || "Unavailable"
              : "Checking TTS status..."}
          </span>
        </div>
      </label>

      <div className="pt-4 mt-2 border-t border-defense-border flex flex-col gap-2">
        <span className="text-xs text-defense-muted uppercase">Hardware Status</span>
        <div className="flex items-center gap-2">
          {modelOk === null ? (
            <span className="text-defense-muted">CHECKING...</span>
          ) : modelOk ? (
            <span className="text-green-500 flex items-center gap-2"><span className="w-2 h-2 bg-green-500 rounded-full inline-block"></span> NPU ONNX LOADED</span>
          ) : (
            <span className="text-red-500 flex items-center gap-2"><span className="w-2 h-2 bg-red-500 rounded-full inline-block"></span> MISSING MODELS</span>
          )}
        </div>
      </div>
    </div>
  );
}
