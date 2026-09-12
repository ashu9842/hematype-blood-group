import { useEffect, useRef, useState } from "react";
import Uploader from "./components/Uploader";
import Results from "./components/Results";
import { predict, API_BASE, type PredictionResult } from "./lib/predict";
import { exportReport } from "./lib/report";

export default function App() {
  const [fingerprint, setFingerprint] = useState<File | null>(null);
  const [smear, setSmear] = useState<File | null>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const fpUrlRef = useRef<string | null>(null);
  const smUrlRef = useRef<string | null>(null);

  useEffect(() => {
    if (fpUrlRef.current) URL.revokeObjectURL(fpUrlRef.current);
    fpUrlRef.current = fingerprint ? URL.createObjectURL(fingerprint) : null;
  }, [fingerprint]);

  useEffect(() => {
    if (smUrlRef.current) URL.revokeObjectURL(smUrlRef.current);
    smUrlRef.current = smear ? URL.createObjectURL(smear) : null;
  }, [smear]);

  useEffect(() => {
    return () => {
      if (fpUrlRef.current) URL.revokeObjectURL(fpUrlRef.current);
      if (smUrlRef.current) URL.revokeObjectURL(smUrlRef.current);
    };
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => setBackendOnline(r.ok))
      .catch(() => setBackendOnline(false));
  }, []);

  const previews = {
    fingerprint: fpUrlRef.current,
    smear: smUrlRef.current,
  };

  const ready = !!fingerprint && !!smear && !scanning;

  async function run() {
    if (!fingerprint || !smear) return;
    setError(null);
    setScanning(true);
    setResult(null);
    try {
      const [res] = await Promise.all([
        predict(fingerprint, smear),
        new Promise((r) => setTimeout(r, 1400)),
      ]);
      setResult(res);
    } catch (e) {
      if (e instanceof Error && e.message === "low_confidence") {
        setError(
          "Invalid input — model could not detect valid biometric features. Please upload a fingerprint scan and a blood smear image."
        );
      } else {
        setError("Inference failed. Check the backend service and retry.");
      }
    } finally {
      setScanning(false);
    }
  }

  return (
    <div className="min-h-full flex flex-col bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border">
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 border border-accent flex items-center justify-center">
              <span className="font-display font-900 text-accent text-sm">H</span>
            </div>
            <div className="leading-tight">
              <div className="font-display font-800 tracking-wide text-sm">
                HEMATYPE<span className="text-accent">/8</span>
              </div>
              <div className="font-mono text-[10px] text-muted tracking-widest">
                MULTI-MODAL ABO/Rh ESTIMATOR
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono text-[10px] text-muted">
            <span
              className={`h-1.5 w-1.5 rounded-full animate-pulse-dot ${
                backendOnline === false ? "bg-accent" : "bg-ok"
              }`}
            />
            {backendOnline === false ? "BACKEND OFFLINE" : "INFERENCE ENGINE ONLINE"}
          </div>
        </div>
      </header>

      {/* Hero / control strip */}
      <div className="border-b border-border relative overflow-hidden">
        <div className="absolute inset-0 grid-overlay opacity-[0.15] pointer-events-none" />
        <div className="mx-auto max-w-6xl px-6 py-10 relative">
          <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr] items-end">
            <div>
              <h1 className="font-display font-900 text-4xl md:text-5xl leading-[0.95] tracking-tight">
                Dual-input blood group
                <br />
                <span className="text-accent">estimation</span>, class by class.
              </h1>
              <p className="mt-4 text-muted max-w-xl leading-relaxed">
                A two-stream deep network fuses a fingerprint biometric (CLAHE
                ridge enhancement) with a microscopic blood-smear /
                hemagglutination image (RGB normalization + thresholding) and
                soft-votes across all eight ABO/Rh phenotypes.
              </p>
            </div>
            <div className="grid grid-cols-4 gap-px bg-border border border-border">
              {[
                ["ARCH", "2-stream"],
                ["FUSION", "late / soft-vote"],
                ["BACKBONE", "EffNet-V2S"],
                ["CLASSES", "8"],
              ].map(([k, v]) => (
                <div key={k} className="bg-panel p-3">
                  <div className="font-mono text-[9px] text-muted tracking-widest">{k}</div>
                  <div className="font-display font-700 text-sm mt-1">{v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Reference image strip */}
      <div className="border-b border-border bg-panel">
        <div className="mx-auto max-w-6xl px-6 py-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="font-mono text-[9px] tracking-widest text-muted">REFERENCE SAMPLES</span>
            <div className="h-px flex-1 bg-border" />
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              {
                url: "https://images.unsplash.com/photo-1576086213369-97a306d36557?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&w=400&q=80",
                label: "BLOOD SMEAR · RBC",
                tag: "STREAM-B INPUT",
              },
              {
                url: "https://images.unsplash.com/photo-1716833322855-e10d5ee0022e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&w=400&q=80",
                label: "LEUKOCYTE SURVEY",
                tag: "AGGLUTINATION",
              },
              {
                url: "https://images.unsplash.com/photo-1611330500121-d9439ddc3d9d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&w=400&q=80",
                label: "FINGERPRINT RIDGE",
                tag: "STREAM-A INPUT",
              },
            ].map(({ url, label, tag }) => (
              <div key={label} className="relative group overflow-hidden border border-border">
                <img
                  src={url}
                  alt={label}
                  className="w-full h-36 object-cover grayscale opacity-60 group-hover:opacity-80 group-hover:grayscale-0 transition-all duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/30 to-transparent" />
                <div className="absolute bottom-0 left-0 p-2.5">
                  <div className="font-mono text-[9px] text-accent tracking-widest">{tag}</div>
                  <div className="font-display font-700 text-[11px] text-foreground mt-0.5">{label}</div>
                </div>
                <div className="absolute top-2 right-2 h-1.5 w-1.5 bg-accent rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main console */}
      <main className="mx-auto max-w-6xl w-full px-6 py-10 flex-1">
        <div className="grid gap-8 lg:grid-cols-[0.85fr_1.15fr]">
          {/* Inputs */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <span className="font-mono text-[10px] tracking-widest text-muted">
                01 — ACQUISITION
              </span>
              <div className="h-px flex-1 bg-border" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Uploader
                id="fp"
                index="A"
                label="Fingerprint"
                hint="Drop ridge scan · SOCOFing-style"
                file={fingerprint}
                onFile={setFingerprint}
                scanning={scanning}
              />
              <Uploader
                id="sm"
                index="B"
                label="Blood Smear"
                hint="Drop smear / agglutination slide"
                file={smear}
                onFile={setSmear}
                scanning={scanning}
              />
            </div>

            <button
              type="button"
              onClick={run}
              disabled={!ready}
              className={`mt-5 w-full py-4 font-display font-700 tracking-widest text-sm transition-colors ${
                ready
                  ? "bg-accent text-white hover:bg-[#c8112e]"
                  : "bg-panel-2 text-muted cursor-not-allowed border border-border"
              }`}
            >
              {scanning
                ? "ANALYZING STREAMS…"
                : ready
                  ? "RUN INFERENCE →"
                  : "LOAD BOTH INPUTS"}
            </button>
            {error && (
              <p className="mt-3 font-mono text-[11px] text-warn leading-relaxed">{error}</p>
            )}

            <p className="mt-4 font-mono text-[10px] leading-relaxed text-muted">
              Both inputs are validated before inference — random or irrelevant
              images are automatically rejected. No biometric data is stored or
              transmitted beyond the local inference session.
            </p>
          </section>

          {/* Output */}
          <section aria-live="polite">
            <div className="flex items-center gap-2 mb-4">
              <span className="font-mono text-[10px] tracking-widest text-muted">
                02 — READOUT
              </span>
              <div className="h-px flex-1 bg-border" />
            </div>

            {result ? (
              <Results
                result={result}
                onExport={() => exportReport(result, previews)}
              />
            ) : (
              <div className="border border-border bg-panel h-full min-h-[360px] flex flex-col items-center justify-center text-center p-8">
                <div className="relative h-16 w-16 border border-border-strong grid-overlay mb-5">
                  {scanning && (
                    <div className="absolute inset-x-0 top-0 h-1/3 bg-gradient-to-b from-accent/70 to-transparent animate-scan" />
                  )}
                </div>
                <p className="font-mono text-xs text-muted max-w-xs leading-relaxed">
                  {scanning
                    ? "Fusing fingerprint + smear feature maps…"
                    : "Awaiting both inputs. The 8-class probability distribution will render here."}
                </p>
              </div>
            )}
          </section>
        </div>

        {/* Disclaimer */}
        <div className="mt-12 border border-accent-dim/60 bg-accent-dim/10 p-5">
          <div className="flex items-start gap-3">
            <span className="font-display font-900 text-accent text-lg leading-none">!</span>
            <p className="font-mono text-[11px] leading-relaxed text-foreground/80">
              <span className="text-accent font-700">
                RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE.
              </span>{" "}
              Predicting blood group from fingerprint biometrics has no
              established scientific or clinical validity. This application is a
              technical demonstration of a multi-modal deep-learning pipeline.
              Outputs must never inform transfusion, cross-matching, or any
              medical decision. Blood typing must be performed by an accredited
              laboratory using serological methods.
            </p>
          </div>
        </div>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-6xl px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-2 font-mono text-[10px] text-muted">
          <span>HEMATYPE/8 · v1.0.0 · FastAPI + PyTorch · EfficientNetV2-S</span>
          <span className="flex items-center gap-1.5">
            <span className="text-border-strong">—</span>
            <span>
              Developed &amp; Designed by{" "}
              <span className="text-accent font-700 tracking-widest">ASHUTOSH</span>
            </span>
            <span className="text-border-strong">·</span>
            <span>All Rights Reserved © {new Date().getFullYear()}</span>
          </span>
        </div>
      </footer>
    </div>
  );
}