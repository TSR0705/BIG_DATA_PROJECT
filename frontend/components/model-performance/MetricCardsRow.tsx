"use client";

import React from "react";
import { BarChart3, CircleDot, Layers, Target, TrendingDown } from "lucide-react";

export interface MetricItem {
  name: string;
  value: string;
  subtext: string;
  icon: React.ElementType;
}

const DEFAULT_METRICS: MetricItem[] = [
  {
    name: "ROC-AUC",
    value: "0.7909",
    subtext: "Validation split (N=46,127)",
    icon: Target,
  },
  {
    name: "PR-AUC",
    value: "0.2853",
    subtext: "Minority default class",
    icon: BarChart3,
  },
  {
    name: "Brier Score",
    value: "0.0654",
    subtext: "Lower is better",
    icon: Layers,
  },
  {
    name: "ECE (Calibrated)",
    value: "0.0042",
    subtext: "Phase 11 OOF Calibrator",
    icon: CircleDot,
  },
  {
    name: "Log Loss",
    value: "0.2353",
    subtext: "Cross-Entropy (Lower is better)",
    icon: TrendingDown,
  },
];

export function MetricCardsRow({ metrics = DEFAULT_METRICS }: { metrics?: MetricItem[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {metrics.map((m) => {
        const Icon = m.icon;
        return (
          <div
            key={m.name}
            className="cl-card p-4 sm:p-5 flex flex-col justify-between bg-white hover:border-[#D5DDD7] transition-all"
          >
            <div className="flex items-center gap-2 mb-2">
              <Icon className="w-5 h-5 text-[#158444] stroke-[2.2] shrink-0" />
              <span className="text-[12px] font-semibold text-[#111613] truncate">
                {m.name}
              </span>
            </div>
            <div>
              <p className="text-[22px] sm:text-[24px] font-extrabold text-[#111613] tracking-tight tabular-nums leading-none">
                {m.value}
              </p>
              <p className="text-[11px] text-[#555E57] mt-1.5 leading-tight truncate">
                {m.subtext}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
