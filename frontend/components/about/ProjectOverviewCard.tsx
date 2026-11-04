"use client";

import React from "react";
import { Target } from "lucide-react";

export function ProjectOverviewCard() {
  return (
    <article className="cl-card p-5 sm:p-6 bg-[#EAF5EE] border border-[#D5EEDB] flex flex-col justify-start">
      <div className="flex items-center gap-2">
        <Target className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Project Overview
        </h2>
      </div>
      <p className="text-[12.5px] sm:text-[13px] text-[#222924] mt-2.5 leading-relaxed">
        CreditLens is a Big Data-driven system that predicts the likelihood of loan default
        using machine learning and provides clear, explainable insights to support fair and
        data-driven lending decisions.
      </p>
    </article>
  );
}
