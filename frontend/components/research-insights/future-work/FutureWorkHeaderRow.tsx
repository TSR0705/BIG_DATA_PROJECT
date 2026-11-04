"use client";

import React from "react";
import { Info, Rocket } from "lucide-react";

export function FutureWorkHeaderRow() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 items-stretch">
      {/* Left Card: Overview */}
      <article className="cl-card p-5 sm:p-6 bg-white flex items-start gap-3.5 lg:col-span-7">
        <Rocket className="w-6 h-6 text-[#158444] shrink-0 mt-0.5 stroke-[2.2]" />
        <div className="min-w-0">
          <h2 className="text-[16px] sm:text-[17px] font-bold text-[#111613] tracking-tight">
            Overview
          </h2>
          <p className="text-[12.5px] sm:text-[13px] text-[#555E57] mt-1 leading-relaxed">
            Planned improvements and research directions to build a more accurate, fair,
            and real-world deployable loan risk prediction system.
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
            Extend the current work with richer data, advanced models, fairness evaluation,
            and real-time, production-ready deployment.
          </p>
        </div>
      </article>
    </div>
  );
}
