"use client";

import React from "react";
import { BarChart3, Check, CircleDot, Database, Layers, Shield } from "lucide-react";

export function KeyFindingsRow() {
  return (
    <article className="cl-card p-5 lg:p-6 bg-white">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2.5">
          <CircleDot className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Key Findings</h2>
        </div>

        <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-[#EAF5EE] text-[#158444] text-[12px] font-semibold">
          <Check className="w-3.5 h-3.5 stroke-[2.8]" />
          <span>Research Summary</span>
        </div>
      </div>

      {/* 4 Highlight Tiles */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
        {/* Tile 1: XGBoost (Highlighted) */}
        <div className="bg-[#EAF5EE] rounded-2xl p-4.5 flex items-start gap-3.5 shadow-2xs">
          <Layers className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h3 className="text-[15px] font-bold text-[#111613] leading-snug">
              XGBoost
            </h3>
            <p className="text-[11.5px] text-[#555E57] mt-1 leading-snug">
              Best performing model for credit risk prediction
            </p>
          </div>
        </div>

        {/* Tile 2: 0.7909 ROC-AUC */}
        <div className="bg-[#FAFBF9] border border-[#EEF2EE] rounded-2xl p-4.5 flex items-start gap-3.5">
          <BarChart3 className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-[20px] font-extrabold text-[#111613] tabular-nums leading-none">
                0.7909
              </span>
              <span className="text-[11.5px] font-bold text-[#555E57]">
                ROC-AUC
              </span>
            </div>
            <p className="text-[11px] text-[#89938D] mt-1 leading-snug">
              Strong ability to distinguish defaulters
            </p>
          </div>
        </div>

        {/* Tile 3: Loan Amount & Age */}
        <div className="bg-[#FAFBF9] border border-[#EEF2EE] rounded-2xl p-4.5 flex items-start gap-3.5">
          <Shield className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h3 className="text-[15px] font-bold text-[#111613] leading-snug">
              Loan Amount &amp; Age
            </h3>
            <p className="text-[11.5px] text-[#555E57] mt-1 leading-snug">
              Most influential risk factors
            </p>
          </div>
        </div>

        {/* Tile 4: Big Data (Spark) */}
        <div className="bg-[#FAFBF9] border border-[#EEF2EE] rounded-2xl p-4.5 flex items-start gap-3.5">
          <Database className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h3 className="text-[15px] font-bold text-[#111613] leading-snug">
              Big Data (Spark)
            </h3>
            <p className="text-[11.5px] text-[#555E57] mt-1 leading-snug">
              Scalable processing of 300K+ loan applications
            </p>
          </div>
        </div>
      </div>
    </article>
  );
}
