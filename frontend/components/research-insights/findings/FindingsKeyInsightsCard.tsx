"use client";

import React from "react";
import { Check, Lightbulb } from "lucide-react";

const FINDINGS_LIST = [
  {
    lead: "XGBoost",
    text: " achieved the best overall performance (ROC-AUC: 0.7909).",
  },
  {
    lead: "Loan Amount",
    text: " has the highest impact on default risk (SHAP: +0.076).",
  },
  {
    lead: "Age",
    text: " is the second most important factor (SHAP: +0.041), with higher risk for younger applicants.",
  },
  {
    lead: "",
    text: "The model is well-calibrated (ECE: 0.0042), indicating reliable probability estimates.",
  },
  {
    lead: "",
    text: "The approach effectively distinguishes between defaulters and non-defaulters on large, imbalanced data.",
  },
];

export function FindingsKeyInsightsCard() {
  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Lightbulb className="w-5 h-5 text-[#158444] shrink-0" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Key Findings
        </h2>
      </div>

      {/* Findings List */}
      <div className="mt-4 space-y-3.5 sm:space-y-4">
        {FINDINGS_LIST.map((item, idx) => (
          <div key={idx} className="flex items-start gap-3">
            <span className="w-5 h-5 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
              <Check className="w-3 h-3 stroke-[3]" />
            </span>
            <p className="text-[12.5px] text-[#222924] leading-relaxed">
              {item.lead ? (
                <>
                  <strong className="font-bold text-[#111613]">{item.lead}</strong>
                  {item.text}
                </>
              ) : (
                item.text
              )}
            </p>
          </div>
        ))}
      </div>
    </article>
  );
}
