"use client";

import React from "react";
import { CircleDot } from "lucide-react";

export function CalibrationCurveCard({ ece = 0.0042 }: { ece?: number }) {
  // SVG Geometry
  const width = 310;
  const height = 210;
  const left = 38;
  const right = 288;
  const top = 18;
  const bottom = 170;

  const plotW = right - left; // 250
  const plotH = bottom - top; // 152

  // Calibration curve deciles: [predicted, observed]
  const calibPoints: [number, number][] = [
    [0.0, 0.0],
    [0.12, 0.11],
    [0.22, 0.21],
    [0.34, 0.33],
    [0.48, 0.47],
    [0.61, 0.6],
    [0.73, 0.72],
    [0.85, 0.84],
    [0.98, 0.97],
  ];

  function getX(pred: number) {
    return left + pred * plotW;
  }
  function getY(obs: number) {
    return bottom - obs * plotH;
  }

  const pathD = calibPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p[0])} ${getY(p[1])}`)
    .join(" ");

  const ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

  return (
    <article className="cl-card p-5 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-2.5 mb-2">
          <CircleDot className="w-5 h-5 text-[#158444] stroke-[2.3] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[16px] font-bold text-[#111613] leading-snug">
              Calibration Curve
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              How close the predicted probabilities are to actual outcomes.
            </p>
          </div>
        </div>

        {/* SVG Plot */}
        <div className="w-full relative mt-3 select-none">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
            {/* Y-Axis Label */}
            <text
              x={-((top + bottom) / 2)}
              y={10}
              transform="rotate(-90)"
              textAnchor="middle"
              className="text-[9.5px] font-medium fill-[#89938D]"
            >
              Observed Probability
            </text>

            {/* X-Axis Label */}
            <text
              x={(left + right) / 2}
              y={height - 2}
              textAnchor="middle"
              className="text-[9.5px] font-medium fill-[#89938D]"
            >
              Predicted Probability
            </text>

            {/* Gridlines & Ticks */}
            {ticks.map((t) => {
              const y = getY(t);
              const x = getX(t);
              return (
                <g key={t}>
                  <line
                    x1={left}
                    y1={y}
                    x2={right}
                    y2={y}
                    stroke="#EEF2EE"
                    strokeWidth="1"
                  />
                  <text
                    x={left - 5}
                    y={y + 3}
                    textAnchor="end"
                    className="text-[9px] font-medium fill-[#89938D]"
                  >
                    {t.toFixed(1)}
                  </text>
                  <text
                    x={x}
                    y={bottom + 12}
                    textAnchor="middle"
                    className="text-[9px] font-medium fill-[#89938D]"
                  >
                    {t.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Perfect Calibration diagonal (Dashed reference) */}
            <line
              x1={left}
              y1={bottom}
              x2={right}
              y2={top}
              stroke="#94A3B8"
              strokeWidth="1.4"
              strokeDasharray="4 4"
            />

            {/* Model Calibrated Curve Line */}
            <path
              d={pathD}
              fill="none"
              stroke="#158444"
              strokeWidth="2.2"
              strokeLinecap="round"
            />

            {/* Calibration Decile Dots */}
            {calibPoints.map((p, i) => (
              <circle
                key={i}
                cx={getX(p[0])}
                cy={getY(p[1])}
                r="3.5"
                fill="#158444"
                stroke="#ffffff"
                strokeWidth="1"
              />
            ))}

            {/* Bottom-Right Legend */}
            <g transform={`translate(${right - 120}, ${bottom - 46})`}>
              {/* Row 1: Model with dot */}
              <line x1="0" y1="6" x2="18" y2="6" stroke="#158444" strokeWidth="2" />
              <circle cx="9" cy="6" r="3" fill="#158444" />
              <text x="24" y="9.5" className="text-[9px] font-semibold fill-[#111613]">
                Model (ECE = {ece.toFixed(4)})
              </text>

              {/* Row 2: Perfect Calibration */}
              <line
                x1="0"
                y1="20"
                x2="18"
                y2="20"
                stroke="#94A3B8"
                strokeWidth="1.5"
                strokeDasharray="3 3"
              />
              <text x="24" y="23.5" className="text-[9px] font-medium fill-[#555E57]">
                Perfect Calibration
              </text>
            </g>
          </svg>
        </div>
      </div>
    </article>
  );
}
