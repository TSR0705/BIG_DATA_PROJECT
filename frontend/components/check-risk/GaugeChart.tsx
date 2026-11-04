"use client";

import React from "react";

interface GaugeProps {
  percentage: number;
}

export function GaugeChart({ percentage }: GaugeProps) {
  // Semicircle geometry:
  // Center (115, 95), Radius: 80, StrokeWidth: 14
  // Semicircle arc starts at left (35, 95) and ends at right (195, 95)
  const r = 80;
  const cx = 115;
  const cy = 95;
  const arcLength = Math.PI * r; // ~251.33

  // Scale: 14.2% maps to ~44% of the semicircle arc (around 11 o'clock)
  const maxScale = 32.0;
  const clampedPct = Math.max(0, Math.min(100, percentage));
  // Ensure that even for small non-zero values (like 2.7%), a graceful visible arc curve is shown
  const rawFraction = clampedPct / maxScale;
  const fillFraction = clampedPct > 0 ? Math.min(1.0, Math.max(0.08, rawFraction)) : 0;
  const strokeDashoffset = arcLength * (1 - fillFraction);

  return (
    <div className="flex flex-col items-center justify-center select-none w-full">
      <div className="w-[230px] h-[125px] relative flex items-center justify-center">
        <svg viewBox="0 0 230 125" className="w-full h-full overflow-visible">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10543A" />
              <stop offset="55%" stopColor="#158444" />
              <stop offset="100%" stopColor="#34D399" />
            </linearGradient>
          </defs>

          {/* Background track (Semicircle) */}
          <path
            d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
            fill="none"
            stroke="#EEF2EE"
            strokeWidth="14"
            strokeLinecap="round"
          />

          {/* Active meter track */}
          {clampedPct > 0 && (
            <path
              d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
              fill="none"
              stroke="url(#gaugeGradient)"
              strokeWidth="14"
              strokeLinecap="round"
              strokeDasharray={`${arcLength} ${arcLength}`}
              strokeDashoffset={strokeDashoffset}
              className="transition-all duration-700 ease-out"
            />
          )}

          {/* Value text: mathematically centered high inside the arch opening */}
          <text
            x={cx}
            y={cy - 22}
            textAnchor="middle"
            className="text-[34px] font-black fill-[#111613] tracking-tight select-none"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {percentage.toFixed(1)}%
          </text>

          {/* Caption text: positioned safely above the arc bottom with zero collision */}
          <text
            x={cx}
            y={cy - 5}
            textAnchor="middle"
            className="text-[10.5px] font-medium fill-[#7E8B82] select-none"
          >
            Estimated Default Probability
          </text>
        </svg>
      </div>
    </div>
  );
}
