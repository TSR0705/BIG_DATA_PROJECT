"use client";

import React from "react";
import { BarChart3, Check, Cpu } from "lucide-react";

const POINTS = [
  "Gradient boosting on decision trees",
  "Handles large, imbalanced datasets",
  "Better accuracy and generalization",
  "Interpretable using SHAP",
];

export function ModelingApproachCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <BarChart3 className="w-6 h-6 text-[#158444] stroke-[2.4] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Modeling Approach
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Machine learning models are trained and compared to find the most effective approach for loan default prediction.
            </p>
          </div>
        </div>

        {/* Content Layout: Left Box + Right 4 Points */}
        <div className="flex flex-col sm:flex-row items-center sm:items-stretch gap-4 mt-5">
          {/* Left XGBoost Box */}
          <div className="w-full sm:w-32 bg-[#EAF5EE] rounded-2xl p-4 flex flex-col items-center justify-center text-center shrink-0 shadow-2xs">
            <Cpu className="w-7 h-7 text-[#158444] stroke-[2] mb-1.5" />
            <h3 className="text-[15px] font-bold text-[#111613] leading-tight">
              XGBoost
            </h3>
            <p className="text-[11px] font-medium text-[#555E57] mt-0.5">
              (Final Model)
            </p>
          </div>

          {/* Right Checklist */}
          <div className="flex-1 space-y-2.5 flex flex-col justify-center">
            {POINTS.map((pt, i) => (
              <div key={i} className="flex items-start gap-2.5">
                <div className="w-4 h-4 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5">
                  <Check className="w-2.5 h-2.5 stroke-[3]" />
                </div>
                <p className="text-[12.5px] text-[#111613] font-medium leading-snug">
                  {pt}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </article>
  );
}
