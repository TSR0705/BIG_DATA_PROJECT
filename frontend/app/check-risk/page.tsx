"use client";

import React, { useState } from "react";
import { ApplicantDetailsForm } from "@/components/check-risk/ApplicantDetailsForm";
import { PredictionResultCard } from "@/components/check-risk/PredictionResultCard";
import { RiskFactorsCard } from "@/components/check-risk/RiskFactorsCard";
import { WhatDoesThisMeanCard } from "@/components/check-risk/WhatDoesThisMeanCard";
import { predictLoanRisk, fetchExampleApplicant } from "@/lib/api";
import type { PredictRiskInput, PredictRiskResult } from "@/types";

const BENCHMARK_INPUT: PredictRiskInput = {
  age: 32,
  annual_income: 850000,
  employment_type: "Salaried",
  loan_amount: 1200000,
  loan_tenure_months: 60,
  existing_obligations: 18000,
};

const BENCHMARK_RESULT: PredictRiskResult = {
  applicant_id: "APP-105781",
  default_probability: 0.142,
  default_percentage: 14.2,
  risk_tier: "Medium Risk",
  risk_category: "Medium",
  confidence: "Demonstration Only",
  model_used: "Illustrative Heuristic Estimate",
  decision: "APPROVE",
  summary:
    "Illustrative heuristic estimate: 14.2% default likelihood based on 6 input ratios (demonstration only, not a validated credit risk assessment).",
  inference_type: "ILLUSTRATIVE_HEURISTIC",
  explanation_type: "ILLUSTRATIVE_HEURISTIC_FACTORS",
  disclaimer:
    "This is an illustrative heuristic estimate based on 6 core input ratios for demonstration purposes only. It is NOT the research-validated 256-feature calibrated XGBoost prediction and should not be used as a validated lending-risk probability.",
  risk_factors: [
    { name: "Loan Amount", impact: 0.076, direction: "+" },
    { name: "Age", impact: 0.041, direction: "+" },
    { name: "Annual Income", impact: 0.038, direction: "-" },
    { name: "Employment Type", impact: 0.026, direction: "-" },
    { name: "Loan Tenure", impact: 0.021, direction: "-" },
  ],
};

export default function CheckRiskPage() {
  const [inputValues, setInputValues] = useState<PredictRiskInput>(BENCHMARK_INPUT);
  const [result, setResult] = useState<PredictRiskResult>(BENCHMARK_RESULT);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function handleSubmit(values: PredictRiskInput) {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await predictLoanRisk(values);
      setResult(res);
      setInputValues(values);
      if (typeof window !== "undefined") {
        sessionStorage.setItem("creditlens_current_prediction", JSON.stringify(res));
      }
    } catch (err: unknown) {
      console.error("Error predicting loan risk:", err);
      const msg = err instanceof Error ? err.message : "Failed to calculate prediction from model backend.";
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadExample() {
    setLoading(true);
    setErrorMsg(null);
    try {
      const example = await fetchExampleApplicant();
      setInputValues(example);
      const res = await predictLoanRisk(example);
      setResult(res);
      if (typeof window !== "undefined") {
        sessionStorage.setItem("creditlens_current_prediction", JSON.stringify(res));
      }
    } catch {
      // Fallback to frozen benchmark values
      setInputValues(BENCHMARK_INPUT);
      setResult(BENCHMARK_RESULT);
      if (typeof window !== "undefined") {
        sessionStorage.setItem("creditlens_current_prediction", JSON.stringify(BENCHMARK_RESULT));
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Title & Subtitle Area */}
      <div>
        <h1 className="text-2xl sm:text-[26px] font-extrabold text-ink tracking-tight">
          Check Loan Risk
        </h1>
        <p className="text-[13px] text-[#555E57] mt-1">
          Enter an applicant&apos;s details to predict the probability of loan default and understand the key risk factors.
        </p>
      </div>

      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs font-medium flex items-center justify-between">
          <span>{errorMsg}</span>
          <button
            onClick={() => setErrorMsg(null)}
            className="text-red-500 hover:text-red-700 font-bold ml-3"
          >
            ×
          </button>
        </div>
      )}

      {/* 4-Card 2x2 Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
        {/* Card 1: Applicant Details */}
        <ApplicantDetailsForm
          initialValues={inputValues}
          onSubmit={handleSubmit}
          onLoadExample={handleLoadExample}
          loading={loading}
        />

        {/* Card 2: Prediction Result */}
        <PredictionResultCard result={result} />

        {/* Card 3: Key Risk Factors */}
        <RiskFactorsCard factors={result.risk_factors} />

        {/* Card 4: What Does This Mean? */}
        <WhatDoesThisMeanCard result={result} />
      </div>
    </div>
  );
}
