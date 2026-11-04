"use client";

import React from "react";
import { BarChart3 } from "lucide-react";

interface FeatureImpactItem {
  name: string;
  subName?: string;
  value: number;
}

const DEFAULT_DATA: FeatureImpactItem[] = [
  { name: "Loan", subName: "Amount", value: 0.076 },
  { name: "Age", value: 0.041 },
  { name: "Annual", subName: "Income", value: -0.038 },
  { name: "Employment", subName: "Type", value: -0.026 },
  { name: "Loan", subName: "Tenure", value: -0.021 },
];

export function FeatureImpactChartCard({
  data = DEFAULT_DATA,
  explanationType = "ILLUSTRATIVE_HEURISTIC_FACTORS",
}: {
  data?: FeatureImpactItem[];
  explanationType?: string;
}) {
  const isTreeShap = explanationType === "GENUINE_TREESHAP";
  // Chart geometry
  const width = 420;
  const height = 230;
  const plotLeft = 52;
  const plotRight = 405;
  const plotTop = 22;
  const plotBottom = 175;

  const yTicks = [
    { label: "0.10", val: 0.1 },
    { label: "0.08", val: 0.08 },
    { label: "0.06", val: 0.06 },
    { label: "0.04", val: 0.04 },
    { label: "0.02", val: 0.02 },
    { label: "0.00", val: 0.0 },
    { label: "-0.02", val: -0.02 },
    { label: "-0.04", val: -0.04 },
    { label: "-0.06", val: -0.06 },
  ];

  // Scale: 0.10 is plotTop, -0.06 is plotBottom
  // Total span = 0.16 units across (plotBottom - plotTop) pixels
  const totalRange = 0.16;
  const plotH = plotBottom - plotTop;
  const zeroY = plotTop + ((0.1 - 0.0) / totalRange) * plotH; // ~117.6px

  function getY(v: number): number {
    return plotTop + ((0.1 - v) / totalRange) * plotH;
  }

  // 5 bar centers
  const barWidth = 36;
  const plotW = plotRight - plotLeft;
  const colStep = plotW / data.length;

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="w-6 h-6 text-[#158444] stroke-[2.4] shrink-0" />
            <h2 className="text-[17px] font-bold text-[#111613]">
              {isTreeShap ? "Feature Impact on Prediction (TreeSHAP)" : "Feature Impact (Illustrative Heuristic)"}
            </h2>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-[11px] font-medium text-[#111613]">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-xs bg-[#10543A]" />
              <span>Increases Risk</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-xs bg-[#86EFAC]" />
              <span>Decreases Risk</span>
            </div>
          </div>
        </div>

        {/* SVG Diverging Bar Chart */}
        <div className="w-full relative mt-2 select-none">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto overflow-visible"
            style={{ maxHeight: "250px" }}
          >
            {/* Y-axis Title (Rotated) */}
            <text
              x={-((plotTop + plotBottom) / 2)}
              y={13}
              transform="rotate(-90)"
              textAnchor="middle"
              className="text-[10px] font-medium fill-[#89938D]"
            >
              {isTreeShap ? "SHAP Value" : "Impact on Risk"}
            </text>

            {/* Gridlines and Ticks */}
            {yTicks.map((t) => {
              const y = getY(t.val);
              const isZero = t.val === 0.0;
              return (
                <g key={t.label}>
                  <line
                    x1={plotLeft}
                    y1={y}
                    x2={plotRight}
                    y2={y}
                    stroke={isZero ? "#89938D" : "#EEF2EE"}
                    strokeWidth={isZero ? 1.2 : 1}
                    strokeDasharray={isZero ? undefined : undefined}
                  />
                  <text
                    x={plotLeft - 6}
                    y={y + 3}
                    textAnchor="end"
                    className={`text-[9.5px] font-medium ${isZero ? "fill-[#111613]" : "fill-[#89938D]"}`}
                  >
                    {t.label}
                  </text>
                </g>
              );
            })}

            {/* Bars and Value Labels */}
            {data.map((item, index) => {
              const cx = plotLeft + colStep * (index + 0.5);
              const isPositive = item.value >= 0;
              const barH = Math.abs(getY(item.value) - zeroY);
              const barY = isPositive ? zeroY - barH : zeroY;

              // Color mapping matching image
              const barColor =
                index === 0
                  ? "#10543A"
                  : index === 1
                  ? "#20865D"
                  : index === 4
                  ? "#BBF7D0"
                  : "#86EFAC";

              const labelY = isPositive ? barY - 5 : barY + barH + 11;
              const sign = isPositive ? "+" : "";

              return (
                <g key={item.name + index}>
                  {/* Bar */}
                  <rect
                    x={cx - barWidth / 2}
                    y={barY}
                    width={barWidth}
                    height={Math.max(2, barH)}
                    fill={barColor}
                    rx={isPositive ? 3 : 3}
                    className="transition-all duration-500"
                  />

                  {/* Value label on top / bottom of bar */}
                  <text
                    x={cx}
                    y={labelY}
                    textAnchor="middle"
                    className="text-[10.5px] font-bold fill-[#111613]"
                  >
                    {sign}{item.value.toFixed(3)}
                  </text>

                  {/* X-axis labels below the chart */}
                  <text
                    x={cx}
                    y={plotBottom + 16}
                    textAnchor="middle"
                    className="text-[10px] font-medium fill-[#111613]"
                  >
                    {item.name}
                  </text>
                  {item.subName && (
                    <text
                      x={cx}
                      y={plotBottom + 27}
                      textAnchor="middle"
                      className="text-[10px] font-medium fill-[#111613]"
                    >
                      {item.subName}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </article>
  );
}
