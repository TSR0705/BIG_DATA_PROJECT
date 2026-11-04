"use client";

import React from "react";
import { Info } from "lucide-react";

export function ProjectOutcomeBanner() {
  return (
    <article className="cl-card p-4 sm:p-5 bg-white border border-[#E5EFE7] flex flex-col sm:flex-row sm:items-center gap-3.5 sm:gap-5">
      {/* Left Title with Info Icon */}
      <div className="flex items-center gap-2.5 shrink-0">
        <Info className="w-6 h-6 text-[#158444] shrink-0 stroke-[2.2]" />
        <h3 className="text-[16px] font-bold text-[#111613] tracking-tight">
          Project Outcome
        </h3>
      </div>

      {/* Vertical Divider (Desktop) */}
      <div className="hidden sm:block w-px h-8 bg-[#D9DFD9] shrink-0" />

      {/* Right Content Text */}
      <p className="text-[12.5px] sm:text-[13px] text-[#222924] leading-relaxed">
        A research-grade, end-to-end, explainable loan default prediction system that
        combines scalable data processing, high-performing machine learning, and transparent
        risk explanations.
      </p>
    </article>
  );
}
