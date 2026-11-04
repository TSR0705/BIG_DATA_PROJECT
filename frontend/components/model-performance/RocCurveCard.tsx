"use client";

import React from "react";
import { TrendingUp } from "lucide-react";

export function RocCurveCard({ auc = 0.7909 }: { auc?: number }) {
  // SVG Geometry
  const width = 310;
  const height = 210;
  const left = 38;
  const right = 288;
  const top = 18;
  const bottom = 170;

  const plotW = right - left; // 250
  const plotH = bottom - top; // 152

  // ROC Curve path points: [FPR, TPR]
  const rocPoints: [number, number][] = [
    [0.0, 0.0],
    [0.02, 0.22],
    [0.05, 0.38],
    [0.1, 0.51],
    [0.18, 0.63],
    [0.3, 0.74],
    [0.45, 0.83],
    [0.65, 0.91],
    [0.85, 0.97],
    [1.0, 1.0],
  ];

  function getX(fpr: number) {
    return left + fpr * plotW;
  }
  function getY(tpr: number) {
    return bottom - tpr * plotH;
  }

  // Construct SVG line and area paths
  const pathD = rocPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p[0])} ${getY(p[1])}`)
    .join(" ");

  const areaD = `${pathD} L ${right} ${bottom} L ${left} ${bottom} Z`;

  const ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

  return (
    <article className="cl-card p-5 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-2.5 mb-2">
          <TrendingUp className="w-5 h-5 text-[#158444] stroke-[2.3] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[16px] font-bold text-[#111613] leading-snug">ROC Curve</h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Ability to distinguish between default and non-default.
            </p>
          </div>
        </div>

        {/* SVG Plot */}
        <div className="w-full relative mt-3 select-none">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
            <defs>
              <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#48C774" stopOpacity="0.32" />
                <stop offset="100%" stopColor="#48C774" stopOpacity="0.03" />
              </linearGradient>
            </defs>

            {/* Y-Axis Label */}
            <text
              x={-((top + bottom) / 2)}
              y={10}
              transform="rotate(-90)"
              textAnchor="middle"
              className="text-[9.5px] font-medium fill-[#89938D]"
            >
              True Positive Rate
            </text>

            {/* X-Axis Label */}
            <text
              x={(left + right) / 2}
              y={height - 2}
              textAnchor="middle"
              className="text-[9.5px] font-medium fill-[#89938D]"
            >
              False Positive Rate
            </text>

            {/* Gridlines & Ticks */}
            {ticks.map((t) => {
              const y = getY(t);
              const x = getX(t);
              return (
                <g key={t}>
                  {/* Horizontal gridline */}
                  <line
                    x1={left}
                    y1={y}
                    x2={right}
                    y2={y}
                    stroke="#EEF2EE"
                    strokeWidth="1"
                  />
                  {/* Y-tick label */}
                  <text
                    x={left - 5}
                    y={y + 3}
                    textAnchor="end"
                    className="text-[9px] font-medium fill-[#89938D]"
                  >
                    {t.toFixed(1)}
                  </text>
                  {/* X-tick label */}
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

            {/* Diagonal Reference line (Random guess) */}
            <line
              x1={left}
              y1={bottom}
              x2={right}
              y2={top}
              stroke="#A7D4B6"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />

            {/* Shaded Area under Curve */}
            <path d={areaD} fill="url(#rocGradient)" />

            {/* ROC Curve line */}
            <path
              d={pathD}
              fill="none"
              stroke="#158444"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* AUC Pill Badge */}
            <g transform={`translate(${right - 88}, ${bottom - 36})`}>
              <rect
                width="84"
                height="24"
                rx="12"
                fill="#EAF5EE"
                stroke="#C6F2D2"
                strokeWidth="1"
              />
              <text
                x="42"
                y="16"
                textAnchor="middle"
                className="text-[10.5px] font-bold fill-[#158444]"
              >
                AUC = {auc.toFixed(4)}
              </text>
            </g>
          </svg>
        </div>
      </div>
    </article>
  );
}
