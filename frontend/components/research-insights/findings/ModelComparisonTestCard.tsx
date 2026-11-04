"use client";

import React from "react";
import { BarChart3, Check } from "lucide-react";

interface ModelMetricRow {
  name: string;
  isOurs?: boolean;
  rocAuc: string;
  prAuc: string;
  brier: string;
  ece: string;
}

const MODELS_DATA: ModelMetricRow[] = [
  {
    name: "XGBoost (Ours)",
    isOurs: true,
    rocAuc: "0.7909",
    prAuc: "0.2853",
    brier: "0.0654",
    ece: "0.0042",
  },
  {
    name: "LightGBM",
    rocAuc: "0.7881",
    prAuc: "0.2819",
    brier: "0.0668",
    ece: "0.0056",
  },
  {
    name: "Random Forest",
    rocAuc: "0.7796",
    prAuc: "0.2558",
    brier: "0.0712",
    ece: "0.0078",
  },
  {
    name: "Logistic Regression",
    rocAuc: "0.7482",
    prAuc: "0.2337",
    brier: "0.0789",
    ece: "0.0104",
  },
];

export function ModelComparisonTestCard() {
  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-[#158444] shrink-0" />
          <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
            Model Comparison (Benchmark)
          </h2>
        </div>
        <p className="text-[12px] text-[#555E57] mt-0.5">
          Phase 11 independent evaluation benchmark (Validation partition, N=46,127).
        </p>
      </div>

      {/* Table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#F0F2F0] text-[11.5px] font-semibold text-[#555E57]">
              <th className="py-2.5 px-3 text-left">Model</th>
              <th className="py-2.5 px-3 text-right">ROC-AUC ↑</th>
              <th className="py-2.5 px-3 text-right">PR-AUC ↑</th>
              <th className="py-2.5 px-3 text-right">Brier ↓</th>
              <th className="py-2.5 px-3 text-right">ECE ↓</th>
            </tr>
          </thead>
          <tbody className="text-[12.5px] divide-y divide-[#F5F7F5]">
            {MODELS_DATA.map((row) => (
              <tr
                key={row.name}
                className={
                  row.isOurs
                    ? "bg-[#EAF5EE] text-[#111613] font-bold rounded-lg"
                    : "text-[#222924] font-medium hover:bg-[#FAFCFA]"
                }
              >
                <td className="py-3 px-3 text-left">
                  <div className="flex items-center gap-1.5">
                    {row.isOurs && (
                      <span className="w-4 h-4 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0">
                        <Check className="w-2.5 h-2.5 stroke-[3]" />
                      </span>
                    )}
                    <span className={row.isOurs ? "font-bold text-[#111613]" : "font-medium"}>
                      {row.name}
                    </span>
                  </div>
                </td>
                <td
                  className={`py-3 px-3 text-right tabular-nums ${
                    row.isOurs ? "font-bold text-[#111613]" : "text-[#333C35]"
                  }`}
                >
                  {row.rocAuc}
                </td>
                <td
                  className={`py-3 px-3 text-right tabular-nums ${
                    row.isOurs ? "font-bold text-[#111613]" : "text-[#333C35]"
                  }`}
                >
                  {row.prAuc}
                </td>
                <td
                  className={`py-3 px-3 text-right tabular-nums ${
                    row.isOurs ? "font-bold text-[#111613]" : "text-[#333C35]"
                  }`}
                >
                  {row.brier}
                </td>
                <td
                  className={`py-3 px-3 text-right tabular-nums ${
                    row.isOurs ? "font-bold text-[#111613]" : "text-[#333C35]"
                  }`}
                >
                  {row.ece}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}
