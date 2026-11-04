"use client";

import React from "react";
import { Check, Layers } from "lucide-react";

const CONTRIBUTIONS = [
  "Built a scalable loan default prediction pipeline using Apache Spark.",
  "Compared multiple ML models and selected an optimized model (XGBoost).",
  "Applied SHAP for explainable and transparent predictions.",
  "Provided actionable insights for risk assessment in lending.",
];

export function OurContributionsCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2.5 mb-4">
          <Layers className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Our Contributions</h2>
        </div>

        {/* 4 Points */}
        <div className="space-y-3.5 mt-2">
          {CONTRIBUTIONS.map((text, i) => (
            <div key={i} className="flex items-start gap-3">
              <div className="w-5 h-5 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
                <Check className="w-3 h-3 stroke-[3]" />
              </div>
              <p className="text-[12.5px] sm:text-[13px] text-[#1F2923] font-medium leading-relaxed">
                {text}
              </p>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}
