"use client";

import React from "react";
import { BarChart3 } from "lucide-react";

export function PrCurveCard({ prAuc = 0.2853 }: { prAuc?: number }) {
  // SVG Geometry
  const width = 310;
  const height = 210;
  const left = 38;
  const right = 288;
  const top = 18;
  const bottom = 170;

  const plotW = right - left; // 250
  const plotH = bottom - top; // 152

  // PR Curve points: [Recall, Precision]
  const prPoints: [number, number][] = [
    [0.0, 1.0],
    [0.02, 0.72],
    [0.04, 0.58],
    [0.08, 0.48],
    [0.15, 0.39],
    [0.25, 0.35],
    [0.4, 0.30],
    [0.6, 0.23],
    [0.8, 0.16],
    [1.0, 0.09],
  ];

  function getX(recall: number) {
    return left + recall * plotW;
  }
  function getY(precision: number) {
    return bottom - precision * plotH;
  }

  const pathD = prPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p[0])} ${getY(p[1])}`)
    .join(" ");

  const areaD = `${pathD} L ${right} ${bottom} L ${left} ${bottom} Z`;

  const ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

  return (
    <article className="cl-card p-5 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-2.5 mb-2">
          <BarChart3 className="w-5 h-5 text-[#158444] stroke-[2.3] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[16px] font-bold text-[#111613] leading-snug">
              Precision–Recall Curve
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Model performance on the minority (default) class.
            </p>
          </div>
        </div>

        {/* SVG Plot */}
        <div className="w-full relative mt-3 select-none">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
            <defs>
              <linearGradient id="prGradient" x1="0%" y1="0%" x2="0%" y2="1">
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
              Precision
            </text>

            {/* X-Axis Label */}
            <text
              x={(left + right) / 2}
              y={height - 2}
              textAnchor="middle"
              className="text-[9.5px] font-medium fill-[#89938D]"
            >
              Recall
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

            {/* Shaded Area under Curve */}
            <path d={areaD} fill="url(#prGradient)" />

            {/* PR Curve line */}
            <path
              d={pathD}
              fill="none"
              stroke="#158444"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* PR-AUC Pill Badge */}
            <g transform={`translate(${right - 106}, ${bottom - 36})`}>
              <rect
                width="102"
                height="24"
                rx="12"
                fill="#EAF5EE"
                stroke="#C6F2D2"
                strokeWidth="1"
              />
              <text
                x="51"
                y="16"
                textAnchor="middle"
                className="text-[10.5px] font-bold fill-[#158444]"
              >
                PR-AUC = {prAuc.toFixed(4)}
              </text>
            </g>
          </svg>
        </div>
      </div>
    </article>
  );
}
