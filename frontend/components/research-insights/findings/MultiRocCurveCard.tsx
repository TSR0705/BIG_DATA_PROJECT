"use client";

import React from "react";
import { TrendingUp } from "lucide-react";

export function MultiRocCurveCard() {
  // Chart geometry
  const width = 460;
  const height = 270;
  const padding = { top: 20, right: 25, bottom: 42, left: 45 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Coordinate helper: (fpr, tpr) where fpr in [0, 1], tpr in [0, 1]
  const getX = (fpr: number) => padding.left + fpr * plotWidth;
  const getY = (tpr: number) => padding.top + (1 - tpr) * plotHeight;

  // Build SVG path from array of [fpr, tpr]
  const buildPath = (points: [number, number][]) => {
    return points
      .map(([fpr, tpr], i) => `${i === 0 ? "M" : "L"} ${getX(fpr).toFixed(1)} ${getY(tpr).toFixed(1)}`)
      .join(" ");
  };

  // Curves data modeled after test-set evaluation results
  const pointsXGB: [number, number][] = [
    [0.0, 0.0],
    [0.01, 0.22],
    [0.03, 0.38],
    [0.06, 0.52],
    [0.1, 0.63],
    [0.15, 0.71],
    [0.2, 0.77],
    [0.3, 0.83],
    [0.4, 0.88],
    [0.5, 0.91],
    [0.6, 0.94],
    [0.7, 0.96],
    [0.8, 0.98],
    [1.0, 1.0],
  ];

  const pointsLGBM: [number, number][] = [
    [0.0, 0.0],
    [0.01, 0.20],
    [0.03, 0.35],
    [0.06, 0.49],
    [0.1, 0.60],
    [0.15, 0.69],
    [0.2, 0.75],
    [0.3, 0.81],
    [0.4, 0.86],
    [0.5, 0.89],
    [0.6, 0.92],
    [0.7, 0.95],
    [0.8, 0.97],
    [1.0, 1.0],
  ];

  const pointsRF: [number, number][] = [
    [0.0, 0.0],
    [0.01, 0.16],
    [0.04, 0.30],
    [0.08, 0.44],
    [0.12, 0.54],
    [0.18, 0.64],
    [0.25, 0.71],
    [0.35, 0.78],
    [0.45, 0.83],
    [0.55, 0.87],
    [0.65, 0.90],
    [0.75, 0.93],
    [0.85, 0.96],
    [1.0, 1.0],
  ];

  const pointsLR: [number, number][] = [
    [0.0, 0.0],
    [0.05, 0.28],
    [0.1, 0.42],
    [0.18, 0.54],
    [0.28, 0.64],
    [0.4, 0.72],
    [0.5, 0.78],
    [0.6, 0.83],
    [0.7, 0.88],
    [0.8, 0.92],
    [1.0, 1.0],
  ];

  const ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-[#158444] shrink-0" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          ROC Curve
        </h2>
      </div>

      {/* SVG Plot */}
      <div className="w-full mt-3 flex items-center justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto max-h-[290px] select-none"
        >
          {/* Grid lines */}
          {ticks.map((t) => (
            <React.Fragment key={t}>
              {/* Horizontal grid line */}
              <line
                x1={getX(0)}
                y1={getY(t)}
                x2={getX(1)}
                y2={getY(t)}
                stroke="#E2E8F0"
                strokeDasharray="3 3"
                strokeWidth="1"
              />
              {/* Vertical grid line */}
              <line
                x1={getX(t)}
                y1={getY(0)}
                x2={getX(t)}
                y2={getY(1)}
                stroke="#E2E8F0"
                strokeDasharray="3 3"
                strokeWidth="1"
              />
              {/* Y-axis label */}
              <text
                x={getX(0) - 8}
                y={getY(t) + 4}
                textAnchor="end"
                fontSize="10"
                fill="#555E57"
                className="font-medium tabular-nums"
              >
                {t.toFixed(1)}
              </text>
              {/* X-axis label */}
              <text
                x={getX(t)}
                y={getY(0) + 16}
                textAnchor="middle"
                fontSize="10"
                fill="#555E57"
                className="font-medium tabular-nums"
              >
                {t.toFixed(1)}
              </text>
            </React.Fragment>
          ))}

          {/* Diagonal Reference Line (y = x) */}
          <line
            x1={getX(0)}
            y1={getY(0)}
            x2={getX(1)}
            y2={getY(1)}
            stroke="#94A3B8"
            strokeDasharray="4 4"
            strokeWidth="1.2"
          />

          {/* Curves */}
          {/* 4. Logistic Regression */}
          <path
            d={buildPath(pointsLR)}
            fill="none"
            stroke="#94A3B8"
            strokeWidth="2"
            strokeDasharray="2 3"
          />

          {/* 3. Random Forest */}
          <path
            d={buildPath(pointsRF)}
            fill="none"
            stroke="#86EFAC"
            strokeWidth="2.2"
            strokeDasharray="6 3 2 3"
          />

          {/* 2. LightGBM */}
          <path
            d={buildPath(pointsLGBM)}
            fill="none"
            stroke="#48C774"
            strokeWidth="2.2"
            strokeDasharray="5 3"
          />

          {/* 1. XGBoost (Ours) - Prominent Solid Line */}
          <path
            d={buildPath(pointsXGB)}
            fill="none"
            stroke="#158444"
            strokeWidth="2.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Axes Titles */}
          <text
            x={-height / 2 + 10}
            y={14}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize="10.5"
            fill="#555E57"
            className="font-semibold"
          >
            True Positive Rate
          </text>
          <text
            x={padding.left + plotWidth / 2}
            y={height - 6}
            textAnchor="middle"
            fontSize="10.5"
            fill="#555E57"
            className="font-semibold"
          >
            False Positive Rate
          </text>

          {/* In-chart Legend (Bottom Right Quadrant) */}
          <g transform={`translate(${getX(0.48)}, ${getY(0.38)})`}>
            <rect
              x="-6"
              y="-10"
              width="180"
              height="80"
              fill="rgba(255, 255, 255, 0.88)"
              rx="6"
              stroke="#EAF0EA"
              strokeWidth="0.8"
            />
            {/* XGBoost */}
            <line x1="0" y1="4" x2="18" y2="4" stroke="#158444" strokeWidth="2.5" />
            <text x="24" y="8" fontSize="10" fill="#111613" fontWeight="600">
              XGBoost (AUC = 0.7909)
            </text>

            {/* LightGBM */}
            <line
              x1="0"
              y1="22"
              x2="18"
              y2="22"
              stroke="#48C774"
              strokeWidth="2"
              strokeDasharray="4 2"
            />
            <text x="24" y="26" fontSize="10" fill="#333C35" fontWeight="500">
              LightGBM (AUC = 0.7881)
            </text>

            {/* Random Forest */}
            <line
              x1="0"
              y1="40"
              x2="18"
              y2="40"
              stroke="#86EFAC"
              strokeWidth="2"
              strokeDasharray="5 2 2 2"
            />
            <text x="24" y="44" fontSize="10" fill="#333C35" fontWeight="500">
              Random Forest (AUC = 0.7796)
            </text>

            {/* Logistic Regression */}
            <line
              x1="0"
              y1="58"
              x2="18"
              y2="58"
              stroke="#94A3B8"
              strokeWidth="2"
              strokeDasharray="2 2"
            />
            <text x="24" y="62" fontSize="10" fill="#555E57" fontWeight="500">
              Logistic Regression (AUC = 0.7482)
            </text>
          </g>
        </svg>
      </div>
    </article>
  );
}
