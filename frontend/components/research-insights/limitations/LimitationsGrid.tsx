"use client";

import React from "react";
import { BarChart3, Check, Database, Layers, Rocket, Scale, Settings } from "lucide-react";

interface LimitationGroup {
  title: string;
  icon: React.ElementType;
  items: string[];
}

const LIMITATION_GROUPS: LimitationGroup[] = [
  {
    title: "Data Limitations",
    icon: Database,
    items: [
      "Single-source dataset (Home Credit).",
      "May not fully represent other regions, periods, or lending policies.",
      "Potential hidden bias in applicant data (e.g., socio-economic factors).",
    ],
  },
  {
    title: "Model Limitations",
    icon: Layers,
    items: [
      "Performance can drop on unseen or out-of-distribution data.",
      "May not capture complex temporal behaviour (e.g., changing income or payment patterns).",
      "Trained on a static snapshot, not real-time data.",
    ],
  },
  {
    title: "Feature Limitations",
    icon: Settings,
    items: [
      "Relies on engineered features from the available dataset.",
      "Lacks richer behavioural and alternative data (e.g., transaction history, device data, credit bureau).",
      "Some relevant risk factors may be unobserved (e.g., customer intent, macroeconomic changes).",
    ],
  },
  {
    title: "Fairness & Bias",
    icon: Scale,
    items: [
      "Risk of bias toward certain demographic or socio-economic groups.",
      "Not explicitly evaluated for fairness metrics (e.g., demographic parity, equal opportunity).",
    ],
  },
  {
    title: "Deployment Limitations",
    icon: Rocket,
    items: [
      "Currently a research prototype, not a production-ready system.",
      "Does not include real-time data ingestion, monitoring, or model retraining.",
      "Scalability and latency are not evaluated in a live environment.",
    ],
  },
  {
    title: "Evaluation Scope",
    icon: BarChart3,
    items: [
      "Evaluated mainly using offline metrics (ROC-AUC, PR-AUC, Brier, ECE).",
      "No long-term or real-world A/B testing.",
      "Calibration and interpretability are limited to the available dataset and model (XGBoost).",
    ],
  },
];

export function LimitationsGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 items-stretch">
      {LIMITATION_GROUPS.map((group) => {
        const Icon = group.icon;
        return (
          <article
            key={group.title}
            className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-start"
          >
            {/* Card Header */}
            <div className="flex items-center gap-2.5 pb-3">
              <Icon className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
              <h3 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
                {group.title}
              </h3>
            </div>

            {/* Bullets List */}
            <div className="mt-1 space-y-3">
              {group.items.map((bullet, idx) => (
                <div key={idx} className="flex items-start gap-2.5">
                  <span className="w-4 h-4 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5">
                    <Check className="w-2.5 h-2.5 stroke-[3]" />
                  </span>
                  <p className="text-[12px] sm:text-[12.5px] text-[#222924] leading-relaxed">
                    {bullet}
                  </p>
                </div>
              ))}
            </div>
          </article>
        );
      })}
    </div>
  );
}
