import type { PredictionResult } from "../lib/predict";

function pct(n: number) {
  return (n * 100).toFixed(1);
}

function GaugeRow({
  label,
  value,
  top,
}: {
  label: string;
  value: number;
  top: boolean;
}) {
  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <span
        className={`font-mono text-[10px] sm:text-xs w-8 sm:w-9 shrink-0 ${
          top ? "text-accent font-700" : "text-muted"
        }`}
      >
        {label}
      </span>
      <div className="relative h-5 sm:h-6 flex-1 bg-panel-2 border border-border overflow-hidden">
        <div
          className={`h-full transition-[width] duration-700 ease-out ${
            top ? "bg-accent" : "bg-border-strong"
          }`}
          style={{ width: `${Math.max(value * 100, 1.5)}%` }}
        />
        <span className="absolute inset-y-0 right-2 flex items-center font-mono text-[9px] sm:text-[10px] text-foreground/80">
          {pct(value)}%
        </span>
      </div>
    </div>
  );
}

// Warn only when real API returns near-uniform output (likely not a biometric image).
// Local-estimator results always show — confidence has no quality signal there.
const WARN_THRESHOLD = 1 / 8 + 0.03; // ~0.155

export default function Results({
  result,
  onExport,
}: {
  result: PredictionResult;
  onExport: () => void;
}) {
  const confidenceBand =
    result.confidence >= 0.7
      ? { label: "HIGH", color: "text-ok" }
      : result.confidence >= 0.4
        ? { label: "MODERATE", color: "text-warn" }
        : { label: "LOW", color: "text-accent" };

  const likelySuspect =
    result.source === "api" && result.confidence < WARN_THRESHOLD;

  return (
    <div className="space-y-2">
      {likelySuspect && (
        <div className="border border-warn/50 bg-warn/5 px-4 py-2.5 flex items-center gap-3">
          <span className="font-display font-900 text-warn text-base leading-none">!</span>
          <p className="font-mono text-[10px] text-warn/90 leading-relaxed">
            Very low confidence — model may not have detected valid biometric features.
            Results below are the model&apos;s best guess only.
          </p>
        </div>
      )}
      <div className="flex flex-col md:grid gap-px bg-border md:grid-cols-5 border border-border">
        {/* Primary readout */}
        <div className="md:col-span-2 bg-panel p-4 sm:p-6 flex flex-col justify-between gap-4">
          <div>
            <div className="flex items-center justify-between">
              <span className="font-mono text-[9px] sm:text-[10px] tracking-widest text-muted">
                PREDICTED PHENOTYPE
              </span>
              <span className={`font-mono text-[9px] sm:text-[10px] tracking-widest ${confidenceBand.color}`}>
                ● {confidenceBand.label}
              </span>
            </div>
            <div className="mt-3 sm:mt-4 flex items-end gap-2 sm:gap-3">
              <span className="font-display font-900 leading-none text-[4rem] sm:text-[5.5rem] text-accent">
                {result.top.replace(/[+-]/, "")}
              </span>
              <span className="font-display font-800 text-3xl sm:text-4xl mb-1 sm:mb-2 text-foreground">
                {result.top.slice(-1)}
              </span>
            </div>
            <div className="mt-2 sm:mt-4 font-mono text-xs text-muted">
              confidence ·{" "}
              <span className="text-foreground">{pct(result.confidence)}%</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-px bg-border border border-border">
            <div className="bg-panel-2 p-2 sm:p-3">
              <div className="font-mono text-[9px] sm:text-[10px] text-muted">FP STREAM</div>
              <div className="font-display font-700 text-base sm:text-lg text-foreground">
                {pct(result.modality.fingerprint)}%
              </div>
            </div>
            <div className="bg-panel-2 p-2 sm:p-3">
              <div className="font-mono text-[9px] sm:text-[10px] text-muted">SMEAR STREAM</div>
              <div className="font-display font-700 text-base sm:text-lg text-foreground">
                {pct(result.modality.smear)}%
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between flex-wrap gap-2">
            <span className="font-mono text-[9px] sm:text-[10px] text-muted">
              {result.source === "api" ? "MODEL: ensemble-v1" : "LOCAL ESTIMATOR"} · {result.latencyMs}ms
            </span>
            <button
              type="button"
              onClick={onExport}
              className="font-mono text-[10px] sm:text-[11px] tracking-widest border border-border-strong px-2 sm:px-3 py-1.5 sm:py-2 hover:border-accent hover:text-accent transition-colors"
            >
              EXPORT ↗
            </button>
          </div>
        </div>

        {/* Distribution */}
        <div className="md:col-span-3 bg-panel p-4 sm:p-6">
          <div className="flex items-center justify-between mb-3 sm:mb-4">
            <span className="font-mono text-[9px] sm:text-[10px] tracking-widest text-muted">
              8 ABO/Rh CLASSES
            </span>
            <span className="font-mono text-[9px] sm:text-[10px] text-muted">SOFTMAX</span>
          </div>
          <div className="space-y-1.5 sm:space-y-2">
            {result.distribution.map((d, i) => (
              <GaugeRow
                key={d.group}
                label={d.group}
                value={d.probability}
                top={i === 0}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}