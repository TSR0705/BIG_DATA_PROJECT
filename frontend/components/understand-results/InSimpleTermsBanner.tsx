"use client";

import React from "react";
import { Info } from "lucide-react";

interface Props {
  text?: string;
}

export function InSimpleTermsBanner({
  text = "Certain details, especially a higher loan amount and the applicant's age, increase the chance of default, while a higher annual income and stable employment reduce the risk.",
}: Props) {
  return (
    <section className="bg-[#F4F9F6] border border-[#DEEDE2] rounded-[20px] p-5 sm:px-6 sm:py-4.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      {/* Left side: Icon + Title */}
      <div className="flex items-center gap-3 shrink-0">
        <Info className="w-5 h-5 text-[#158444] stroke-[2.2] shrink-0" />
        <h2 className="text-[16px] font-bold text-[#111613] whitespace-nowrap">
          In Simple Terms
        </h2>
      </div>

      {/* Vertical divider on desktop */}
      <div className="h-8 w-px bg-[#D5E5DA] mx-2 hidden sm:block shrink-0" />

      {/* Right side: Explanatory text */}
      <p className="text-[13px] text-[#1F2923] font-medium leading-relaxed flex-1">
        {text}
      </p>
    </section>
  );
}
