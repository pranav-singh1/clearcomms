import { useCallback } from "react";

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
    <div className="flex flex-col gap-4">
      <div
        className={`border border-dashed transition-colors p-6 flex items-center justify-center text-center font-mono text-sm cursor-pointer ${file ? 'border-defense-accent bg-defense-accent/5 text-white' : 'border-defense-border bg-defense-900 text-defense-muted hover:border-defense-muted'}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept={ACCEPT}
          onChange={handleInput}
          className="hidden"
          id="file-input"
        />
        <label htmlFor="file-input" className="cursor-pointer w-full h-full min-h-[100px] flex items-center justify-center">
          {file ? file.name : "DROP FILE / BROWSE"}
        </label>
      </div>
      <button
        type="button"
        onClick={onTranscribe}
        disabled={disabled || loading}
        className="w-full py-3 bg-white text-black font-semibold text-sm hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-mono uppercase"
      >
        {loading ? "INITIALIZING..." : "EXECUTE"}
      </button>
    </div>
  );
}