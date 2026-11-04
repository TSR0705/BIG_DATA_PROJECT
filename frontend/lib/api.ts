/**
 * Typed client for the CreditLens FastAPI backend.
 * There is deliberately no fallback data here: when the API is unreachable the UI
 * shows an error state instead of invented numbers.
 */
import type {
  ApplicantScore,
  ApplicantSearchHit,
  DefaultRateProfileData,
  ProfileDimension,
  ProjectOverviewData,
  RecentPredictionData,
} from "@/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(`Cannot reach the CreditLens API at ${API_BASE_URL}`, 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(`Cannot reach the CreditLens API at ${API_BASE_URL}`, 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const b = await res.json();
      if (b?.detail) detail = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

export function fetchProjectOverview(profile: ProfileDimension = "age") {
  return getJson<ProjectOverviewData>(`/api/project/overview?profile=${profile}`);
}

export function fetchDefaultRateProfile(dimension: ProfileDimension) {
  return getJson<DefaultRateProfileData>(`/api/project/default-rate-profile?dimension=${dimension}`);
}

export function fetchRecentPredictions(limit = 5) {
  return getJson<RecentPredictionData[]>(`/api/predictions/recent?limit=${limit}`);
}

export function searchApplicants(q: string, limit = 6) {
  return getJson<{ query: string; count: number; results: ApplicantSearchHit[] }>(
    `/api/applicants/search?q=${encodeURIComponent(q)}&limit=${limit}`,
  );
}

export function scoreApplicant(skIdCurr: number) {
  return getJson<ApplicantScore>(`/api/applicants/${skIdCurr}/score`);
}

export function predictLoanRisk(input: import("@/types").PredictRiskInput) {
  return postJson<import("@/types").PredictRiskResult>("/api/applicants/predict-risk", input);
}

export function fetchExampleApplicant() {
  return getJson<import("@/types").PredictRiskInput>("/api/applicants/example");
}

export function apiDocsUrl() {
  return `${API_BASE_URL}/docs`;
}
