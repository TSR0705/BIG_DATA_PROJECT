"use client";

import React from "react";
import { Database, Layers, Users, Zap } from "lucide-react";

export function PipelineKpiRow() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
      {/* Card 1: Total Records */}
      <article className="cl-card p-5 bg-white flex items-center gap-4">
        <span className="w-11 h-11 rounded-xl bg-[#E0F9E6] text-[#158444] flex items-center justify-center shrink-0">
          <Users className="w-5 h-5 stroke-[2.2]" />
        </span>
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Total Records</p>
          <p className="text-[24px] sm:text-[26px] font-extrabold text-[#111613] tracking-tight tabular-nums mt-1 leading-tight">
            307,511
          </p>
          <p className="text-[11.5px] font-medium text-[#158444] mt-0.5">
            ↗ Full dataset (100%)
          </p>
        </div>
      </article>

      {/* Card 2: Total Features */}
      <article className="cl-card p-5 bg-white flex items-center gap-4">
        <span className="w-11 h-11 rounded-xl bg-[#E0F9E6] text-[#158444] flex items-center justify-center shrink-0">
          <Layers className="w-5 h-5 stroke-[2.2]" />
        </span>
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Total Features</p>
          <p className="text-[24px] sm:text-[26px] font-extrabold text-[#111613] tracking-tight tabular-nums mt-1 leading-tight">
            240
          </p>
          <p className="text-[11.5px] font-medium text-[#89938D] mt-0.5">
            After feature engineering
          </p>
        </div>
      </article>

      {/* Card 3: Missing Values */}
      <article className="cl-card p-5 bg-white flex items-center gap-4">
        <span className="w-11 h-11 rounded-xl bg-[#E0F9E6] text-[#158444] flex items-center justify-center shrink-0">
          <Database className="w-5 h-5 stroke-[2.2]" />
        </span>
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Missing Values</p>
          <p className="text-[24px] sm:text-[26px] font-extrabold text-[#111613] tracking-tight tabular-nums mt-1 leading-tight">
            &lt; 1%
          </p>
          <p className="text-[11.5px] font-medium text-[#89938D] mt-0.5">
            After preprocessing
          </p>
        </div>
      </article>

      {/* Card 4: Processing Engine */}
      <article className="cl-card p-5 bg-white flex items-center gap-4">
        <span className="w-11 h-11 rounded-xl bg-[#E0F9E6] text-[#158444] flex items-center justify-center shrink-0">
          <Zap className="w-5 h-5 stroke-[2.2]" />
        </span>
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-[#555E57] leading-none">Processing Engine</p>
          <p className="text-[22px] sm:text-[24px] font-extrabold text-[#111613] tracking-tight truncate mt-1 leading-tight">
            Apache Spark
          </p>
          <p className="text-[11.5px] font-medium text-[#89938D] mt-0.5 truncate">
            Distributed big data processing
          </p>
        </div>
      </article>
    </div>
  );
}
