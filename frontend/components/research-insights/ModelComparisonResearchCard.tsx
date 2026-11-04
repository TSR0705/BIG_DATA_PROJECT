"use client";

import React from "react";
import { BarChart3 } from "lucide-react";

interface ModelItem {
  name: string;
  score: number;
  color: string;
}

const MODELS: ModelItem[] = [
  { name: "XGBoost", score: 0.7909, color: "#10543A" },
  { name: "LightGBM", score: 0.7881, color: "#20865D" },
  { name: "Random Forest", score: 0.7796, color: "#86EFAC" },
  { name: "Logistic Regression", score: 0.7482, color: "#BBF7D0" },
];

export function ModelComparisonResearchCard() {
  const minVal = 0.7;
  const maxVal = 0.8;
  const ticks = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8];

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2.5 mb-5">
          <BarChart3 className="w-6 h-6 text-[#158444] stroke-[2.4] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">
            Model Comparison{" "}
            <span className="font-normal text-[#555E57] text-[15px]">(Research Result)</span>
          </h2>
        </div>

        {/* 4 Horizontal Bars */}
        <div className="space-y-4">
          {MODELS.map((m) => {
            const widthPct = Math.min(100, Math.max(8, ((m.score - minVal) / (maxVal - minVal)) * 100));
            return (
              <div
                key={m.name}
                className="grid grid-cols-[minmax(0,8.5rem)_1fr_auto] items-center gap-3"
              >
                <span className="text-[13px] font-medium text-[#111613] truncate">
                  {m.name}
                </span>

                <div className="h-4 rounded-full bg-[#EEF2EE] overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${widthPct}%`,
                      backgroundColor: m.color,
                    }}
                  />
                </div>

                <span className="text-[13px] font-bold text-[#111613] tabular-nums w-14 text-right">
                  {m.score.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>

        {/* X-Axis with Ticks and Label */}
        <div className="mt-5 pt-3 border-t border-[#EEF2EE]">
          <div className="grid grid-cols-[minmax(0,8.5rem)_1fr_auto] items-center gap-3">
            <div className="w-full" />
            <div className="flex justify-between items-center text-[10.5px] font-medium text-[#89938D] px-1">
              {ticks.map((t) => (
                <span key={t} className="tabular-nums">
                  {t.toFixed(2)}
                </span>
              ))}
            </div>
            <div className="w-14" />
          </div>
          <p className="text-center text-[10.5px] font-medium text-[#555E57] mt-1.5">
            ROC-AUC (Higher is Better)
          </p>
        </div>
      </div>
    </article>
  );
}
