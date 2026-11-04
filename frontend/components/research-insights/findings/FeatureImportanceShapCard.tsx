"use client";

import React from "react";
import { Layers } from "lucide-react";

interface ShapFeature {
  name: string;
  value: number;
  displayValue: string;
  color: string;
}

const SHAP_FEATURES: ShapFeature[] = [
  {
    name: "Loan Amount",
    value: 0.076,
    displayValue: "0.076",
    color: "bg-[#10543A]", // dark forest green
  },
  {
    name: "Age",
    value: 0.041,
    displayValue: "0.041",
    color: "bg-[#20865D]", // medium green
  },
  {
    name: "Annual Income",
    value: 0.038,
    displayValue: "0.038",
    color: "bg-[#48C774]", // vibrant emerald green
  },
  {
    name: "Employment Type",
    value: 0.026,
    displayValue: "0.026",
    color: "bg-[#86EFAC]", // soft mint green
  },
  {
    name: "Loan Tenure",
    value: 0.021,
    displayValue: "0.021",
    color: "bg-[#BBF7D0]", // pale mint green
  },
];

const MAX_VALUE = 0.08;

export function FeatureImportanceShapCard() {
  const ticks = [0.00, 0.02, 0.04, 0.06, 0.08];

  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-[#158444] shrink-0" />
          <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
            Feature Importance (SHAP)
          </h2>
        </div>
        <p className="text-[12px] text-[#555E57] mt-0.5">
          Top features that drive the model&apos;s prediction.
        </p>
      </div>

      {/* Bars List */}
      <div className="mt-5 space-y-3.5">
        {SHAP_FEATURES.map((feat) => {
          const widthPercent = (feat.value / MAX_VALUE) * 100;
          return (
            <div key={feat.name} className="flex items-center gap-3">
              {/* Feature label */}
              <div className="w-28 sm:w-32 shrink-0 text-[12px] font-medium text-[#111613] truncate">
                {feat.name}
              </div>

              {/* Bar track and fill */}
              <div className="flex-1 h-3.5 bg-transparent rounded-full flex items-center">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${feat.color}`}
                  style={{ width: `${Math.min(widthPercent, 100)}%` }}
                />
              </div>

              {/* Numerical value */}
              <div className="w-10 shrink-0 text-right text-[11.5px] font-medium text-[#111613] tabular-nums">
                {feat.displayValue}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom X-Axis */}
      <div className="mt-3 pt-2">
        <div className="flex items-center">
          {/* Offset matching label width */}
          <div className="w-28 sm:w-32 shrink-0" />

          {/* Ticks & Line Container */}
          <div className="flex-1 pr-10">
            {/* Axis horizontal line */}
            <div className="w-full h-px bg-[#D9DFD9]" />

            {/* Ticks row */}
            <div className="flex justify-between text-[10px] text-[#555E57] font-medium mt-1 tabular-nums">
              {ticks.map((t) => (
                <span key={t}>{t.toFixed(2)}</span>
              ))}
            </div>

            {/* Axis label */}
            <p className="text-center text-[11px] font-semibold text-[#555E57] mt-1.5">
              Mean |SHAP Value|
            </p>
          </div>
        </div>
      </div>
    </article>
  );
}
