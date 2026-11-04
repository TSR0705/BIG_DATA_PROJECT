"use client";

import React from "react";
import { BarChart3, Database, Layers, Shield } from "lucide-react";

export function KeyHighlightsCard() {
  const highlights = [
    {
      icon: Database,
      value: "300K+",
      label: "Loan Records",
      isLarge: true,
    },
    {
      icon: Layers,
      value: "XGBoost",
      label: "Best Model",
      isLarge: true,
    },
    {
      icon: BarChart3,
      value: "0.7909",
      label: "ROC-AUC",
      isLarge: true,
    },
    {
      icon: Shield,
      value: "Explainable AI",
      label: "via SHAP",
      isLarge: false,
    },
  ];

  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3.5">
        <BarChart3 className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Key Highlights
        </h2>
      </div>

      {/* 4 KPI Sub-tiles */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-3.5">
        {highlights.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.label}
              className="bg-[#FAFCFA] border border-[#EEF2EE] rounded-xl p-3 text-center flex flex-col items-center justify-center"
            >
              <Icon className="w-5 h-5 text-[#158444] mb-1.5 stroke-[2.2]" />
              <p
                className={`text-[#111613] tracking-tight leading-tight ${
                  item.isLarge
                    ? "text-[18px] sm:text-[20px] font-extrabold tabular-nums"
                    : "text-[13.5px] sm:text-[14.5px] font-bold"
                }`}
              >
                {item.value}
              </p>
              <p className="text-[11px] text-[#555E57] mt-1 truncate">
                {item.label}
              </p>
            </div>
          );
        })}
      </div>
    </article>
  );
}
