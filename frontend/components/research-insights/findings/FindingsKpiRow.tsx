"use client";

import React from "react";
import { BarChart3, Check, Layers, Shield, Target } from "lucide-react";

export function FindingsKpiRow() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
      {/* Card 1: Best Model */}
      <article className="cl-card p-5 bg-white flex items-start gap-3.5">
        <Target className="w-6 h-6 text-[#158444] shrink-0 mt-0.5 stroke-[2.2]" />
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Best Model</p>
          <p className="text-[22px] sm:text-[24px] font-extrabold text-[#111613] tracking-tight mt-1 leading-tight">
            XGBoost
          </p>
          <div className="mt-1.5">
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-[#EAF5EE] text-[#158444] text-[11px] font-semibold">
              <Check className="w-3 h-3 stroke-[2.8]" />
              <span>Final Model</span>
            </span>
          </div>
        </div>
      </article>

      {/* Card 2: ROC-AUC */}
      <article className="cl-card p-5 bg-white flex items-start gap-3.5">
        <BarChart3 className="w-6 h-6 text-[#158444] shrink-0 mt-0.5 stroke-[2.2]" />
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">ROC-AUC</p>
          <p className="text-[22px] sm:text-[24px] font-extrabold text-[#111613] tracking-tight tabular-nums mt-1 leading-tight">
            0.7909
          </p>
          <p className="text-[11.5px] font-medium text-[#158444] mt-1.5 flex items-center gap-0.5">
            <span>↗</span>
            <span>+1.29% vs. LightGBM</span>
          </p>
        </div>
      </article>

      {/* Card 3: Default Rate */}
      <article className="cl-card p-5 bg-white flex items-start gap-3.5">
        <Shield className="w-6 h-6 text-[#158444] shrink-0 mt-0.5 stroke-[2.2]" />
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Default Rate</p>
          <p className="text-[22px] sm:text-[24px] font-extrabold text-[#111613] tracking-tight tabular-nums mt-1 leading-tight">
            8.07%
          </p>
          <p className="text-[11.5px] font-medium text-[#555E57] mt-1.5">
            In the overall dataset
          </p>
        </div>
      </article>

      {/* Card 4: Key Insight */}
      <article className="cl-card p-5 bg-white flex items-start gap-3.5">
        <Layers className="w-6 h-6 text-[#158444] shrink-0 mt-0.5 stroke-[2.2]" />
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Key Insight</p>
          <p className="text-[16px] sm:text-[18px] font-bold text-[#111613] tracking-tight mt-1 leading-snug">
            Loan Amount &amp; Age
          </p>
          <p className="text-[11.5px] font-medium text-[#555E57] mt-1 truncate">
            are the most influential factors
          </p>
        </div>
      </article>
    </div>
  );
}
