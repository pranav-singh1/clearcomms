import "./Controls.css";

type Props = {
  applyRadioFilter: boolean;
  setApplyRadioFilter: (v: boolean) => void;
  normalize: boolean;
  setNormalize: (v: boolean) => void;
  modelOk: boolean | null;
  onCheckModel: () => void;
};

export function Controls({
  applyRadioFilter,
  setApplyRadioFilter,
  normalize,
  setNormalize,
  modelOk,
  onCheckModel,
}: Props) {
  return (
    <div className="controls">
      <h2 className="controls-title">Controls</h2>

      <label className="control-row">
        <input
          type="checkbox"
          checked={applyRadioFilter}
          onChange={(e) => setApplyRadioFilter(e.target.checked)}
        />
        <span>Apply radio preprocess (bandpass + light gate)</span>
      </label>
      <p className="control-hint">Fast DSP filter that often helps radio audio.</p>

      <label className="control-row">
        <input
          type="checkbox"
          checked={normalize}
          onChange={(e) => setNormalize(e.target.checked)}
        />
        <span>Normalize loudness (peak)</span>
      </label>
      <p className="control-hint">Whisper expects 16 kHz mono; we convert automatically.</p>

      <hr className="controls-divider" />

      <h3 className="controls-subtitle">Model backend</h3>
      {modelOk === null ? (
        <button type="button" className="btn btn-ghost" onClick={onCheckModel}>
          Check model status
        </button>
      ) : modelOk ? (
        <p className="status status-ok">ONNX encoder/decoder found → QNN/NPU backend</p>
      ) : (
        <p className="status status-warn">ONNX models not found in ./models</p>
      )}
      <p className="control-hint">Config: config.yaml</p>
    </div>
  );
}
