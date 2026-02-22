import { useCallback } from "react";
import "./UploadZone.css";

const ACCEPT = ".wav,.flac,.ogg,.mp3,.m4a,audio/*";

type Props = {
  file: File | null;
  onFileChange: (f: File | null) => void;
  onTranscribe: () => void;
  loading: boolean;
  disabled: boolean;
};

export function UploadZone({ file, onFileChange, onTranscribe, loading, disabled }: Props) {
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && ACCEPT.split(",").some((ext) => ext.startsWith(".") ? f.name.toLowerCase().endsWith(ext.slice(1)) : true))
        onFileChange(f);
    },
    [onFileChange]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => e.preventDefault(), []);

  const handleInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      onFileChange(f ?? null);
      e.target.value = "";
    },
    [onFileChange]
  );

  return (
    <div className="upload-zone">
      <div
        className={`drop-area ${file ? "has-file" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept={ACCEPT}
          onChange={handleInput}
          className="drop-input"
          id="file-input"
        />
        <label htmlFor="file-input" className="drop-label">
          {file ? (
            <span className="file-name">{file.name}</span>
          ) : (
            <>Drop an audio clip here or click to browse</>
          )}
        </label>
      </div>
      <button
        type="button"
        className="btn btn-primary"
        onClick={onTranscribe}
        disabled={disabled || loading}
      >
        {loading ? "Transcribing…" : "Transcribe"}
      </button>
    </div>
  );
}
