"use client";

import React from "react";
import { MetricCardsRow } from "@/components/model-performance/MetricCardsRow";
import { RocCurveCard } from "@/components/model-performance/RocCurveCard";
import { PrCurveCard } from "@/components/model-performance/PrCurveCard";
import { CalibrationCurveCard } from "@/components/model-performance/CalibrationCurveCard";
import { ModelComparisonCard } from "@/components/model-performance/ModelComparisonCard";
import { KeyInsightsCard } from "@/components/model-performance/KeyInsightsCard";

export default function ModelPerformancePage() {
  return (
    <div className="space-y-5">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl sm:text-[26px] font-extrabold text-[#111613] tracking-tight">
          Model Performance
        </h1>
        <p className="text-[13px] text-[#555E57] mt-1">
          Evaluate how well the model predicts loan default and how reliable its probability estimates are.
        </p>
      </div>

      {/* Row 1: 5 Top Metric KPI Cards */}
      <MetricCardsRow />

      {/* Row 2: 3 Curve Cards (ROC, PR, Calibration) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
        <RocCurveCard auc={0.7909} />
        <PrCurveCard prAuc={0.2853} />
        <CalibrationCurveCard ece={0.0042} />
      </div>

      {/* Row 3: Model Comparison Table (7 cols) + Key Insights (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        <div className="lg:col-span-7">
          <ModelComparisonCard />
        </div>
        <div className="lg:col-span-5">
          <KeyInsightsCard />
        </div>
      </div>
    </div>
  );
}
