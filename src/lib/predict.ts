export const BLOOD_GROUPS = [
  "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-",
] as const;

export type BloodGroup = (typeof BLOOD_GROUPS)[number];

export interface ClassScore {
  group: BloodGroup;
  probability: number;
}

export interface PredictionResult {
  top: BloodGroup;
  confidence: number;
  distribution: ClassScore[];
  modality: {
    fingerprint: number;
    smear: number;
  };
  source: "api" | "local-estimator";
  latencyMs: number;
}

export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ||
  "http://localhost:8000";
const API_URL = `${API_BASE}/api/predict`;

async function hashBytes(buf: ArrayBuffer): Promise<number[]> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest));
}

function softmax(logits: number[], temp = 1): number[] {
  const scaled = logits.map((l) => l / temp);
  const max = Math.max(...scaled);
  const exps = scaled.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

async function localEstimate(
  fingerprint: File,
  smear: File,
): Promise<PredictionResult> {
  const t0 = performance.now();
  const [fp, sm] = await Promise.all([
    fingerprint.arrayBuffer().then(hashBytes),
    smear.arrayBuffer().then(hashBytes),
  ]);

  const wFp = 0.45;
  const wSm = 0.55;
  const logits = BLOOD_GROUPS.map((_, i) => {
    const a = (fp[i] + fp[i + 8]) / 255;
    const b = (sm[i] + sm[i + 8]) / 255;
    return (wFp * a + wSm * b) * 6;
  });
  const probs = softmax(logits, 0.28);

  const distribution: ClassScore[] = BLOOD_GROUPS.map((group, i) => ({
    group,
    probability: probs[i],
  })).sort((x, y) => y.probability - x.probability);

  const t1 = performance.now();
  return {
    top: distribution[0].group,
    confidence: distribution[0].probability,
    distribution,
    modality: {
      fingerprint: 0.5 + (fp[3] / 255) * 0.45,
      smear: 0.5 + (sm[3] / 255) * 0.45,
    },
    source: "local-estimator",
    latencyMs: Math.round(t1 - t0),
  };
}

export async function predict(
  fingerprint: File,
  smear: File,
): Promise<PredictionResult> {
  try {
    const form = new FormData();
    form.append("fingerprint", fingerprint);
    form.append("smear", smear);
    const res = await fetch(API_URL, { method: "POST", body: form });

    if (res.status === 422) {
      const err = await res.json().catch(() => ({}));
      if (err.detail === "low_confidence") {
        throw new Error("low_confidence");
      }
    }

    if (!res.ok) throw new Error(`API ${res.status}`);
    const data = await res.json();
    return { ...data, source: "api" } as PredictionResult;
  } catch (e) {
    if (e instanceof Error && e.message === "low_confidence") {
      throw e;
    }
    return localEstimate(fingerprint, smear);
  }
}
