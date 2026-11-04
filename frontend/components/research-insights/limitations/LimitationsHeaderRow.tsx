"use client";

import React from "react";
import { Info, Target } from "lucide-react";

export function LimitationsHeaderRow() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 items-stretch">
      {/* Left Card: Overview */}
      <article className="cl-card p-5 sm:p-6 bg-white flex items-start gap-3.5 lg:col-span-7">
        <Target className="w-6 h-6 text-[#158444] shrink-0 mt-0.5 stroke-[2.2]" />
        <div className="min-w-0">
          <h2 className="text-[16px] sm:text-[17px] font-bold text-[#111613] tracking-tight">
            Overview
          </h2>
          <p className="text-[12.5px] sm:text-[13px] text-[#555E57] mt-1 leading-relaxed">
            Key limitations of the current study that may affect the model&apos;s
            performance, generalizability, and real-world deployment.
          </p>
        </div>
      </article>

      {/* Right Card: In Short */}
      <article className="cl-card p-5 sm:p-6 bg-[#EAF5EE] border border-[#D5EEDB] flex items-start gap-3.5 lg:col-span-5">
        <div className="w-5 h-5 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5">
          <Info className="w-3.5 h-3.5 stroke-[2.5]" />
        </div>
        <div className="min-w-0">
          <h3 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
            In Short
          </h3>
          <p className="text-[12px] sm:text-[12.5px] text-[#222924] mt-1 leading-relaxed">
            The model shows strong predictive performance, but it still has practical,
            methodological, and data-related limitations.
          </p>
        </div>
      </article>
    </div>
  );
}
