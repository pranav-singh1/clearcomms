import { useState, useEffect } from "react";
import { modelStatus, type TtsStatus } from "../api";

type Props = {
  applyRadioFilter: boolean;
  setApplyRadioFilter: (v: boolean) => void;
  radioIntensity: number;
  setRadioIntensity: (v: number) => void;
  normalize: boolean;
  setNormalize: (v: boolean) => void;
  ttsEnabled: boolean;
  setTtsEnabled: (v: boolean) => void;
  ttsStatus: TtsStatus | null;
  realtimeTtsEnabled: boolean;
  setRealtimeTtsEnabled: (v: boolean) => void;
  micActive: boolean;
};

export function Controls({
  applyRadioFilter,
  setApplyRadioFilter,
  radioIntensity,
  setRadioIntensity,
  normalize,
  setNormalize,
  ttsEnabled,
  setTtsEnabled,
  ttsStatus,
  realtimeTtsEnabled,
  setRealtimeTtsEnabled,
  micActive,
}: Props) {
  const [modelOk, setModelOk] = useState<boolean | null>(null);

  useEffect(() => {
    modelStatus().then(res => setModelOk(res.models_found)).catch(() => setModelOk(false));
  }, []);

  const realtimeDisabledReason = !ttsEnabled
    ? "Enable TTS (online) to use realtime speech."
    : !ttsStatus?.available
    ? ttsStatus?.reason || "TTS backend unavailable."
    : !micActive
    ? "Enable Mic input to use realtime speech."
    : null;
  const realtimeDisabled = Boolean(realtimeDisabledReason);

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

      {applyRadioFilter && (
        <div className="flex flex-col gap-2 pl-7">
          <div className="flex justify-between items-center">
            <span className="text-xs text-defense-muted uppercase tracking-widest">Radio Intensity</span>
            <span className="text-xs text-defense-accent font-mono tabular-nums">
              {radioIntensity === 50 ? "BASELINE" : radioIntensity < 50 ? `MILD (${radioIntensity})` : `HEAVY (${radioIntensity})`}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={radioIntensity}
            onChange={(e) => setRadioIntensity(Number(e.target.value))}
            className="w-full h-1 appearance-none bg-defense-border accent-defense-accent cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-defense-muted font-mono">
            <span>MILD</span>
            <span>BASELINE</span>
            <span>HEAVY</span>
          </div>
        </div>
      )}

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

      <label className={`flex items-start gap-3 ${realtimeDisabled ? "cursor-not-allowed" : "cursor-pointer"} group`}>
        <input
          type="checkbox"
          checked={realtimeTtsEnabled}
          onChange={(e) => setRealtimeTtsEnabled(e.target.checked)}
          disabled={realtimeDisabled}
          title={realtimeDisabledReason || undefined}
          className="mt-1 appearance-none w-4 h-4 border border-defense-border checked:bg-defense-accent checked:border-defense-accent transition-colors relative disabled:opacity-50"
        />
        <div className="flex flex-col gap-1">
          <span className="text-white group-hover:text-defense-accent transition-colors">Realtime TTS (Mic)</span>
          <span className="text-xs text-defense-muted">Speak cleaned transcript segments automatically (online)</span>
          {realtimeDisabledReason && (
            <span className="text-xs text-amber-400">{realtimeDisabledReason}</span>
          )}
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
