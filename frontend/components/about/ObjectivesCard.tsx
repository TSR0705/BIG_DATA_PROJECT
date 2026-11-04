"use client";

import React from "react";
import { Check, Layers } from "lucide-react";

export function ObjectivesCard() {
  const objectives = [
    "Predict loan default risk accurately.",
    "Build a scalable Big Data processing pipeline.",
    "Provide explainable and transparent predictions.",
    "Enable insights for fair and informed lending.",
  ];

  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-start">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3.5">
        <Layers className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Objectives
        </h2>
      </div>

      {/* Bullets List */}
      <div className="space-y-3">
        {objectives.map((obj, idx) => (
          <div key={idx} className="flex items-start gap-2.5">
            <span className="w-4 h-4 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
              <Check className="w-2.5 h-2.5 stroke-[3]" />
            </span>
            <p className="text-[12px] sm:text-[12.5px] text-[#222924] leading-relaxed">
              {obj}
            </p>
          </div>
        ))}
      </div>
    </article>
  );
}
