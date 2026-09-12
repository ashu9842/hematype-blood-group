import { useCallback, useEffect, useRef, useState } from "react";

interface UploaderProps {
  id: string;
  label: string;
  index: string;
  hint: string;
  file: File | null;
  onFile: (file: File | null) => void;
  scanning: boolean;
}

export default function Uploader({
  id,
  label,
  index,
  hint,
  file,
  onFile,
  scanning,
}: UploaderProps) {
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file && preview) {
      URL.revokeObjectURL(preview);
      setPreview(null);
    }
  }, [file]);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, []);

  const accept = useCallback(
    (f: File | null) => {
      if (!f) return;
      if (!f.type.startsWith("image/")) {
        setFileError("Image file chahiye (JPG, PNG, BMP)");
        setTimeout(() => setFileError(null), 3000);
        return;
      }
      if (f.size > 8 * 1024 * 1024) {
        setFileError("File 8MB se badi hai");
        setTimeout(() => setFileError(null), 3000);
        return;
      }
      setFileError(null);
      onFile(f);
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(f);
      });
    },
    [onFile],
  );

  return (
    <div className="flex flex-col">
      <div className="flex items-baseline justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-accent tracking-widest">
            {index}
          </span>
          <span className="font-display font-700 text-sm tracking-wide uppercase">
            {label}
          </span>
        </div>
        <span
          className={`font-mono text-[10px] tracking-widest ${
            file ? "text-ok" : "text-muted"
          }`}
        >
          {file ? "LOADED" : "AWAITING"}
        </span>
      </div>

      {fileError && (
        <p className="mb-1 font-mono text-[10px] text-accent">{fileError}</p>
      )}

      <div
        onDragOver={(e) => { e.preventDefault(); if (!scanning) setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!scanning) accept(e.dataTransfer.files?.[0] ?? null);
        }}
        onClick={() => !scanning && inputRef.current?.click()}
        role="button"
        aria-label={`Upload ${label} image`}
        tabIndex={scanning ? -1 : 0}
        onKeyDown={(e) => {
          if (!scanning && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        className={`group relative aspect-square overflow-hidden border transition-colors ${
          scanning
            ? "cursor-not-allowed opacity-70 border-border bg-panel-2"
            : over
              ? "cursor-pointer border-accent bg-accent-dim/10"
              : "cursor-pointer border-border hover:border-border-strong bg-panel-2"
        }`}
      >
        {preview ? (
          <>
            <img
              src={preview}
              alt={`${label} preview`}
              className="absolute inset-0 h-full w-full object-cover"
            />
            <div className="absolute inset-0 bg-background/30" />
            {scanning && (
              <div className="absolute inset-x-0 top-0 h-1/4 bg-gradient-to-b from-accent/60 to-transparent animate-scan" />
            )}
          </>
        ) : (
          <div className="absolute inset-0 grid-overlay opacity-40" />
        )}

        <div className="relative h-full w-full flex flex-col items-center justify-center gap-2 p-4 text-center">
          {!preview && (
            <>
              <div className="h-10 w-10 border border-border-strong flex items-center justify-center text-muted group-hover:text-accent transition-colors">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                >
                  <path d="M12 16V4m0 0L7 9m5-5 5 5" />
                  <path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
                </svg>
              </div>
              <p className="font-mono text-[11px] text-muted max-w-[80%] leading-relaxed">
                {hint}
              </p>
            </>
          )}
        </div>

        {preview && (
          <div className="absolute bottom-0 inset-x-0 flex items-center justify-between px-3 py-2 bg-background/70 backdrop-blur-sm">
            <span className="font-mono text-[10px] text-foreground/80 truncate max-w-[70%]">
              {file?.name}
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onFile(null);
                setPreview((prev) => {
                  if (prev) URL.revokeObjectURL(prev);
                  return null;
                });
              }}
              className="font-mono text-[10px] text-accent hover:underline"
            >
              CLEAR
            </button>
          </div>
        )}

        <input
          id={id}
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => accept(e.target.files?.[0] ?? null)}
        />
      </div>
    </div>
  );
}
