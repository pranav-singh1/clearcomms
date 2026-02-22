import { useMemo } from "react";
import type { TranscribeResult } from "./api";
import "./Result.css";

type Props = {
  result: TranscribeResult;
  originalFile: File | null;
  applyRadioFilter: boolean;
};

export function Result({ result, originalFile, applyRadioFilter }: Props) {
  const originalUrl = useMemo(() => (originalFile ? URL.createObjectURL(originalFile) : null), [originalFile]);
  const preparedUrl = useMemo(
    () => (result.audio_prepared_b64 ? `data:audio/wav;base64,${result.audio_prepared_b64}` : null),
    [result.audio_prepared_b64]
  );
  const filteredUrl = useMemo(
    () => (result.audio_filtered_b64 ? `data:audio/wav;base64,${result.audio_filtered_b64}` : null),
    [result.audio_filtered_b64]
  );

  const handleDownloadTranscript = () => {
    const blob = new Blob([result.text + "\n"], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "transcript.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handleDownloadMeta = () => {
    const blob = new Blob([JSON.stringify({ meta: result.meta }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "metadata.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const transcriptContent = result.error
    ? `Transcription failed: ${result.error}`
    : (result.text || "(no transcript)");
  const transcriptIsError = Boolean(result.error);

  return (
    <div className="result">
      <div className="result-grid">
        <div className="result-audios">
          <h3 className="result-heading">Original upload</h3>
          {originalUrl ? (
            <audio controls src={originalUrl} className="result-audio" />
          ) : (
            <p className="result-caption">Not available</p>
          )}
          <p className="result-caption">
            Loaded: {result.sample_rate_original} Hz, {result.duration_sec}s
          </p>

          <h3 className="result-heading">Prepared (16 kHz mono)</h3>
          {preparedUrl ? (
            <audio controls src={preparedUrl} className="result-audio" />
          ) : (
            <p className="result-caption">Not available</p>
          )}

          <h3 className="result-heading">Radio preprocess (filtered / noised clip)</h3>
          {applyRadioFilter && filteredUrl ? (
            <audio controls src={filteredUrl} className="result-audio" />
          ) : applyRadioFilter ? (
            <p className="result-caption">Not available</p>
          ) : (
            <p className="result-caption">Enable “Apply radio preprocess” in Controls to generate this clip.</p>
          )}
        </div>

        <div className="result-transcript-block">
          <h3 className="result-heading">Transcript</h3>
          <div className={`transcript-text ${transcriptIsError ? "transcript-error" : ""}`}>
            {transcriptContent}
          </div>

          <h3 className="result-heading">Performance</h3>
          <pre className="result-meta">
            {Object.keys(result.meta).length ? JSON.stringify(result.meta, null, 2) : "—"}
          </pre>

          <h3 className="result-heading">Export</h3>
          <div className="result-actions">
            <button type="button" className="btn btn-ghost" onClick={handleDownloadTranscript}>
              Download transcript.txt
            </button>
            <button type="button" className="btn btn-ghost" onClick={handleDownloadMeta}>
              Download metadata.json
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
