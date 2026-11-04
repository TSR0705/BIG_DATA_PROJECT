"use client";

import React from "react";
import { Layers } from "lucide-react";
import type { RiskFactorImpactData } from "@/types";

interface Props {
  factors?: RiskFactorImpactData[];
  explanationType?: string;
}

const DEFAULT_FACTORS: RiskFactorImpactData[] = [
  { name: "Loan Amount", impact: 0.076, direction: "+" },
  { name: "Age", impact: 0.041, direction: "+" },
  { name: "Annual Income", impact: 0.038, direction: "-" },
  { name: "Employment Type", impact: 0.026, direction: "-" },
  { name: "Loan Tenure", impact: 0.021, direction: "-" },
];

const BAR_COLORS = ["#10543A", "#20865D", "#48C774", "#86EFAC", "#BBF7D0"];

export function WhyThisPredictionCard({
  factors = DEFAULT_FACTORS,
  explanationType = "ILLUSTRATIVE_HEURISTIC_FACTORS",
}: Props) {
  const maxImpact = 0.095;
  const isTreeShap = explanationType === "GENUINE_TREESHAP";

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <Layers className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Why This Prediction?{" "}
              <span className="font-normal text-[#555E57] text-[15px]">
                {isTreeShap ? "(TreeSHAP Analysis)" : "(Illustrative Factors)"}
              </span>
            </h2>
            <p className="text-[12px] text-[#89938D] mt-0.5">
              {isTreeShap
                ? "Exact TreeSHAP attributions from the 256-feature XGBoost model."
                : "Heuristic factor impact based on financial ratios. (Not TreeSHAP — full model requires 256 features)."}
            </p>
          </div>
        </div>

        {/* 5 Progress Bars */}
        <ul className="mt-5 space-y-4">
          {factors.map((f, i) => {
            const widthPct = Math.min(100, Math.max(15, (f.impact / maxImpact) * 100));
            const sign = f.direction === "+" ? "+" : "-";
            return (
              <li
                key={f.name}
                className="grid grid-cols-[minmax(0,8.5rem)_1fr_auto] items-center gap-3"
              >
                <span className="text-[13px] font-medium text-[#111613] truncate">
                  {f.name}
                </span>
                <div className="h-3 rounded-full bg-[#EEF2EE] overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${widthPct}%`,
                      backgroundColor: BAR_COLORS[i % BAR_COLORS.length],
                    }}
                  />
                </div>
                <span className="text-[13px] font-bold text-[#111613] tabular-nums w-14 text-right">
                  {sign}{f.impact.toFixed(3)}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </article>
  );
}
