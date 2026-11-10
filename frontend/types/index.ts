/** Types mirrored from the FastAPI response models (api/app/schemas/project.py). */

export type BadgeType = "green" | "red" | "teal";
export type RiskTier = "Low Risk" | "Medium Risk" | "High Risk";
export type ProfileDimension = "age" | "credit" | "income";

export interface KPICardData {
  title: string;
  value: string;
  numeric_value: number;
  subtitle: string;
  trend: "up" | "down" | "neutral" | string;
  badge_type: BadgeType | string;
  source: string;
  sparkline: number[];
  sparkline_label: string;
}

export interface RiskFactorData {
  rank: number;
  feature: string;
  display_name: string;
  mean_abs_shap: number;
  mean_shap: number;
  direction: "increase" | "decrease" | "categorical" | string;
  rank_correlation: number | null;
}

export interface RecentPredictionData {
  applicant_id: string;
  sk_id_curr: number | null;
  scored_at: string;
  risk_tier: RiskTier | string;
  probability: number;
  probability_percent: string;
  decision: string;
  source: string;
  model_version?: string;
  inference_type?: "GENUINE_MODEL" | "ILLUSTRATIVE_ESTIMATE" | string;
}

export interface ProfileBand {
  band: string;
  lower: number;
  upper: number;
  applicants: number;
  defaults: number;
  default_rate: number;
}

export interface DefaultRateProfileData {
  dimension: ProfileDimension | string;
  label: string;
  axis_title: string;
  feature: string;
  overall_default_rate: number;
  bands: ProfileBand[];
  peak_band: string | null;
  applicants_outside_bands: number;
  source: string;
}

export interface ApplicantDistributionData {
  total: number;
  non_default_count: number;
  non_default_percent: number;
  default_count: number;
  default_percent: number;
}

export interface ProjectOverviewData {
  project_name: string;
  subtitle: string;
  generated_at: string;
  total_applicants: number;
  total_defaults: number;
  default_rate: number;
  non_defaults: number;
  non_default_rate: number;
  model_roc_auc: number;
  model_pr_auc: number;
  model_ece: number;
  brier_score: number;
  model_version: string;
  kpis: KPICardData[];
  distribution: ApplicantDistributionData;
  default_rate_profile: DefaultRateProfileData;
  profile_dimensions: { key: string; label: string }[];
  top_risk_factors: RiskFactorData[];
  shap_context: { sample_size: number; partition: string; units: string; source: string };
  recent_predictions: RecentPredictionData[];
  key_takeaways: string[];
  data_sources: Record<string, string>;
}

export interface ApplicantSearchHit {
  sk_id_curr: number;
  applicant_id: string;
  source: "labelled" | "inference" | string;
}

export interface PredictRiskInput {
  age: number;
  annual_income: number;
  employment_type: string;
  loan_amount: number;
  loan_tenure_months: number;
  existing_obligations: number;
}

export interface RiskFactorImpactData {
  name: string;
  impact: number;
  direction: string;
}

export interface PredictRiskResult {
  applicant_id: string;
  default_probability: number;
  default_percentage: number;
  risk_tier: string;
  risk_category: string;
  confidence: string;
  model_used: string;
  decision: string;
  summary: string;
  risk_factors: RiskFactorImpactData[];
  inference_type?: "GENUINE_MODEL" | "ILLUSTRATIVE_HEURISTIC" | string;
  explanation_type?: "GENUINE_TREESHAP" | "ILLUSTRATIVE_HEURISTIC_FACTORS" | string;
  disclaimer?: string;
}

export interface ApplicantScore {
  applicant_id: string;
  sk_id_curr: number;
  default_probability: number;
  default_percentage: number;
  decision: "APPROVE" | "REJECT" | string;
  binary_prediction: number;
  risk_tier: "LOW" | "MEDIUM" | "HIGH" | string;
  threshold_used: number;
  model_version: string;
  partition: string;
  scored_at: string;
  amt_credit: number | null;
  amt_annuity: number | null;
  amt_income_total: number | null;
  code_gender: string | null;
  age_years: number | null;
  actual_target: number | null;
}
