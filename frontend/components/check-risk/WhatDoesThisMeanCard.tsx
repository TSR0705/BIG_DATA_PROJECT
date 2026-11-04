"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, BarChart3, Info, Lightbulb, ShieldCheck } from "lucide-react";
import type { PredictRiskResult } from "@/types";

interface Props {
  result: PredictRiskResult;
}

export function WhatDoesThisMeanCard({ result }: Props) {
  const probText = `${result.default_percentage.toFixed(1)}% Probability`;
  const riskTierText = result.risk_tier;

  const riskDescription =
    result.risk_tier === "Low Risk"
      ? "The risk is low. The applicant exhibits a healthy credit and financial profile."
      : result.risk_tier === "High Risk"
      ? "The risk is high. Stronger underwriting scrutiny or collateral should be required."
      : "The risk is moderate. Additional checks may be considered.";

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <span className="w-10 h-10 rounded-xl bg-[#E0F9E6] flex items-center justify-center text-[#158444] shrink-0">
            <Info className="w-5 h-5 stroke-[2.2]" />
          </span>
          <h2 className="text-[17px] font-bold text-[#111613]">What Does This Mean?</h2>
        </div>

        {/* 3 Informative Rows */}
        <div className="space-y-5 mt-5">
          {/* Item 1: Probability */}
          <div className="flex items-start gap-3.5">
            <span className="w-9 h-9 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0 mt-0.5">
              <BarChart3 className="w-4 h-4 stroke-[2]" />
            </span>
            <div>
              <h3 className="text-[13.5px] font-bold text-[#111613] leading-snug">{probText}</h3>
              <p className="text-[12px] text-[#555E57] mt-0.5 leading-relaxed">
                Based on similar past applicants, there is a {result.default_percentage.toFixed(1)}% chance of default.
              </p>
            </div>
          </div>

          {/* Item 2: Risk Level */}
          <div className="flex items-start gap-3.5">
            <span className="w-9 h-9 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0 mt-0.5">
              <ShieldCheck className="w-4 h-4 stroke-[2]" />
            </span>
            <div>
              <h3 className="text-[13.5px] font-bold text-[#111613] leading-snug">{riskTierText}</h3>
              <p className="text-[12px] text-[#555E57] mt-0.5 leading-relaxed">
                {riskDescription}
              </p>
            </div>
          </div>

          {/* Item 3: Not a Final Decision */}
          <div className="flex items-start gap-3.5">
            <span className="w-9 h-9 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0 mt-0.5">
              <Lightbulb className="w-4 h-4 stroke-[2]" />
            </span>
            <div>
              <h3 className="text-[13.5px] font-bold text-[#111613] leading-snug">Not a Final Decision</h3>
              <p className="text-[12px] text-[#555E57] mt-0.5 leading-relaxed">
                This is a model estimate to support better, data-driven decisions.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Action CTA link to Understand Results */}
      <div className="pt-4 mt-4 border-t border-[#EEF2EE] flex items-center justify-between">
        <span className="text-[11.5px] text-[#89938D]">
          {result.inference_type === "ILLUSTRATIVE_HEURISTIC" ? "Illustrative heuristic" : "Calibrated model score"}
        </span>
        <Link
          href="/understand-results"
          className="inline-flex items-center gap-1.5 text-[12.5px] font-semibold text-[#158444] hover:text-[#0E6030] transition-colors"
        >
          <span>View Detailed Breakdown</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>
    </article>
  );
}
