"use client";

import React from "react";
import { Check, Lightbulb } from "lucide-react";

const INSIGHTS = [
  "XGBoost achieves the best overall performance (ROC-AUC: 0.7909).",
  "The model is well-calibrated (ECE: 0.0042), so its risk probabilities are reliable.",
  "Performance is strong even with an imbalanced dataset (PR-AUC: 0.2853).",
  "The model effectively distinguishes between default and non-default applicants.",
];

export function KeyInsightsCard({ items = INSIGHTS }: { items?: string[] }) {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2.5 mb-5">
          <Lightbulb className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Key Insights</h2>
        </div>

        {/* 4 Items */}
        <div className="space-y-4">
          {items.map((text, i) => (
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
