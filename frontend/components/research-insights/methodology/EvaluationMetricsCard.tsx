"use client";

import React from "react";
import {
  BarChart3,
  Layers,
  Percent,
  Share2,
  Shield,
  Target,
  TrendingUp,
} from "lucide-react";

interface MetricBadge {
  name: string;
  icon: React.ElementType;
}

const METRICS: MetricBadge[] = [
  { name: "ROC-AUC", icon: TrendingUp },
  { name: "ECE", icon: Shield },
  { name: "PR-AUC", icon: BarChart3 },
  { name: "R² Score", icon: Percent },
  { name: "Brier Score", icon: Layers },
  { name: "SHAP", icon: Share2 },
];

export function EvaluationMetricsCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <Target className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Evaluation Metrics
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Metrics used to evaluate model performance and calibration.
            </p>
          </div>
        </div>

        {/* 6 Badges Grid (2 cols x 3 rows) */}
        <div className="grid grid-cols-2 gap-2.5 mt-4">
          {METRICS.map((m) => {
            const Icon = m.icon;
            return (
              <div
                key={m.name}
                className="bg-[#FAFBF9] border border-[#EEF2EE] rounded-xl px-3.5 py-2.5 flex items-center gap-2.5 hover:border-[#D5DDD7] transition-all"
              >
                <Icon className="w-4 h-4 text-[#158444] stroke-[2.2] shrink-0" />
                <span className="text-[12.5px] font-bold text-[#111613]">
                  {m.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </article>
  );
}
