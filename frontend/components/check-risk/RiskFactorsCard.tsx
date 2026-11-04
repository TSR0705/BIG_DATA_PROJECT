"use client";

import React from "react";
import { Layers } from "lucide-react";
import type { RiskFactorImpactData } from "@/types";

interface Props {
  factors: RiskFactorImpactData[];
}

const BAR_COLORS = ["#10543A", "#20865D", "#48C774", "#86EFAC", "#BBF7D0"];

export function RiskFactorsCard({ factors }: Props) {
  const maxImpact = 0.095;

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <span className="w-10 h-10 rounded-xl bg-[#E0F9E6] flex items-center justify-center text-[#158444] shrink-0">
            <Layers className="w-5 h-5 stroke-[2.2]" />
          </span>
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">Key Risk Factors</h2>
            <p className="text-[12px] text-[#89938D] mt-0.5">
              Top factors that influenced this prediction (SHAP Analysis).
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
