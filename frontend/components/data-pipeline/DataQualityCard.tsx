"use client";

import React from "react";
import { ShieldCheck } from "lucide-react";

export function DataQualityCard() {
  // SVG Donut geometry
  const size = 150;
  const cx = 75;
  const cy = 75;
  const r = 54;
  const strokeWidth = 16;
  const circumference = 2 * Math.PI * r; // ~339.29

  const validPct = 0.992;
  const missingPct = 0.008;

  const validOffset = 0;
  const validDash = validPct * circumference;

  const missingDash = missingPct * circumference;
  const missingOffset = -(validPct * circumference);

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <ShieldCheck className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Data Quality
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Data after preprocessing.
            </p>
          </div>
        </div>

        {/* Donut Chart + Legend */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mt-6 my-auto select-none">
          {/* SVG Donut */}
          <div className="relative w-[150px] h-[150px] shrink-0 flex items-center justify-center">
            <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full -rotate-90">
              {/* Background ring */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="#EEF2EE"
                strokeWidth={strokeWidth}
              />

              {/* Missing data arc (0.8%) */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="#86EFAC"
                strokeWidth={strokeWidth}
                strokeDasharray={`${missingDash} ${circumference}`}
                strokeDashoffset={missingOffset}
              />

              {/* Valid data arc (99.2%) */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="#158444"
                strokeWidth={strokeWidth}
                strokeDasharray={`${validDash} ${circumference}`}
                strokeDashoffset={validOffset}
              />
            </svg>

            {/* Center Label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
              <span className="text-[24px] font-extrabold text-[#111613] tracking-tight leading-none tabular-nums">
                99.2%
              </span>
              <span className="text-[11px] font-medium text-[#89938D] mt-1">
                Valid Data
              </span>
            </div>
          </div>

          {/* Legend */}
          <div className="space-y-3.5">
            <div className="flex items-start gap-2.5">
              <span className="w-3 h-3 rounded-full bg-[#158444] shrink-0 mt-1" />
              <div>
                <p className="text-[13px] font-bold text-[#111613] leading-none">
                  Valid Data
                </p>
                <p className="text-[12px] font-medium text-[#555E57] mt-0.5">
                  99.2%
                </p>
              </div>
            </div>

            <div className="flex items-start gap-2.5">
              <span className="w-3 h-3 rounded-full bg-[#86EFAC] shrink-0 mt-1" />
              <div>
                <p className="text-[13px] font-bold text-[#111613] leading-none">
                  Missing Data
                </p>
                <p className="text-[12px] font-medium text-[#555E57] mt-0.5">
                  0.8%
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </article>
  );
}
