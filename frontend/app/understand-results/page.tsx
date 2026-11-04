"use client";

import React, { useSyncExternalStore, useMemo } from "react";
import { PredictionSummaryCard } from "@/components/understand-results/PredictionSummaryCard";
import { KeyTakeawaysCard } from "@/components/understand-results/KeyTakeawaysCard";
import { WhyThisPredictionCard } from "@/components/understand-results/WhyThisPredictionCard";
import { FeatureImpactChartCard } from "@/components/understand-results/FeatureImpactChartCard";
import { InSimpleTermsBanner } from "@/components/understand-results/InSimpleTermsBanner";
import type { RiskFactorImpactData } from "@/types";

const BENCHMARK_FACTORS: RiskFactorImpactData[] = [
  { name: "Loan Amount", impact: 0.076, direction: "+" },
  { name: "Age", impact: 0.041, direction: "+" },
  { name: "Annual Income", impact: 0.038, direction: "-" },
  { name: "Employment Type", impact: 0.026, direction: "-" },
  { name: "Loan Tenure", impact: 0.021, direction: "-" },
];

function subscribe(callback: () => void) {
  if (typeof window === "undefined") return () => {};
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

function getSnapshot(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return sessionStorage.getItem("creditlens_current_prediction");
  } catch {
    return null;
  }
}

function getServerSnapshot(): string | null {
  return null;
}

export default function UnderstandResultsPage() {
  const storedJson = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const currentPrediction = useMemo(() => {
    if (!storedJson) return null;
    try {
      return JSON.parse(storedJson);
    } catch {
      return null;
    }
  }, [storedJson]);

  const applicantId = currentPrediction?.applicant_id ?? "APP-105781";
  const percentage =
    typeof currentPrediction?.default_percentage === "number"
      ? currentPrediction.default_percentage
      : typeof currentPrediction?.probability === "number"
      ? currentPrediction.probability * 100
      : 14.2;
  const riskTier = currentPrediction?.risk_tier ?? "Medium Risk";
  const riskCategory = currentPrediction?.risk_category ?? "Medium";
  const confidence = currentPrediction?.confidence ?? "Demonstration Only";
  const modelUsed = currentPrediction?.model_used ?? "Illustrative Heuristic Estimate";
  const inferenceType = currentPrediction?.inference_type ?? "ILLUSTRATIVE_HEURISTIC";
  const explanationType = currentPrediction?.explanation_type ?? "ILLUSTRATIVE_HEURISTIC_FACTORS";
  const disclaimer =
    currentPrediction?.disclaimer ??
    "This is an illustrative heuristic estimate based on 6 core input ratios for demonstration purposes only. It is NOT the research-validated 256-feature calibrated XGBoost prediction and should not be used as a validated lending-risk probability.";
  const summary =
    currentPrediction?.summary ??
    "Illustrative heuristic estimate: 14.2% likelihood based on 6 input ratios (demonstration only, not a validated credit risk assessment).";
  const factors: RiskFactorImpactData[] =
    Array.isArray(currentPrediction?.risk_factors) && currentPrediction.risk_factors.length > 0
      ? currentPrediction.risk_factors
      : BENCHMARK_FACTORS;

  // Format factors for FeatureImpactChartCard
  const chartData = factors.map((f) => {
    const val = f.direction === "+" ? f.impact : -f.impact;
    const parts = f.name.split(" ");
    return {
      name: parts[0],
      subName: parts.slice(1).join(" ") || undefined,
      value: val,
    };
  });

  const dynamicTakeaways = [
    {
      title: "Main Reason",
      description: `The prediction is mainly influenced by ${factors[0]?.name.toLowerCase() || "loan amount"} and ${factors[1]?.name.toLowerCase() || "age"}.`,
    },
    {
      title: "Risk Level",
      description: `This is evaluated as ${riskTier.toLowerCase()} (${percentage.toFixed(1)}%). Additional checks may be considered.`,
    },
    {
      title: "Reliability",
      description: "The model is well-calibrated and provides a reliable probability estimate.",
    },
    {
      title: "Next Step",
      description: "Use this insight as a decision support, not a final approval/rejection.",
    },
  ];

  return (
    <div className="space-y-5">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl sm:text-[26px] font-extrabold text-[#111613] tracking-tight">
          Understand Results
        </h1>
        <p className="text-[13px] text-[#555E57] mt-1">
          A detailed explanation of the prediction, key risk factors, and what it means for the applicant.
        </p>
      </div>

      {/* Row 1: Prediction Summary (7 cols) + Key Takeaways (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        <div className="lg:col-span-7">
          <PredictionSummaryCard
            applicantId={applicantId}
            percentage={percentage}
            riskTier={riskTier}
            riskCategory={riskCategory}
            confidence={confidence}
            modelUsed={modelUsed}
            summary={summary}
            inferenceType={inferenceType}
            disclaimer={disclaimer}
          />
        </div>
        <div className="lg:col-span-5">
          <KeyTakeawaysCard items={dynamicTakeaways} />
        </div>
      </div>

      {/* Row 2: Why This Prediction + Feature Impact Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        <div className="lg:col-span-6">
          <WhyThisPredictionCard factors={factors} explanationType={explanationType} />
        </div>
        <div className="lg:col-span-6">
          <FeatureImpactChartCard data={chartData} explanationType={explanationType} />
        </div>
      </div>

      {/* Row 3: In Simple Terms Banner */}
      <InSimpleTermsBanner text={summary || undefined} />
    </div>
  );
}
