"use client";

import React from "react";
import { Lightbulb } from "lucide-react";

export function LimitationsTakeawayBanner() {
  return (
    <article className="cl-card p-4 sm:p-5 bg-white border border-[#E5EFE7] flex flex-col sm:flex-row sm:items-center gap-3.5 sm:gap-5">
      {/* Left Title with Lightbulb Icon */}
      <div className="flex items-center gap-2.5 shrink-0">
        <Lightbulb className="w-6 h-6 text-[#158444] shrink-0 stroke-[2.2]" />
        <h3 className="text-[16px] font-bold text-[#111613] tracking-tight">
          Takeaway
        </h3>
      </div>

      {/* Vertical Divider (Desktop) */}
      <div className="hidden sm:block w-px h-8 bg-[#D9DFD9] shrink-0" />

      {/* Right Content Text */}
      <p className="text-[12.5px] sm:text-[13px] text-[#222924] leading-relaxed">
        These limitations highlight areas for improvement and guide future work toward a
        more robust, fair, and real-world deployable credit risk prediction system.
      </p>
    </article>
  );
}
