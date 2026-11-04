"use client";

import React from "react";
import { Check, Layers } from "lucide-react";

interface ModelBenchmark {
  name: string;
  isOurs?: boolean;
  rocAuc: string;
  prAuc: string;
  brier: string;
  ece: string;
  status: "Selected" | "—";
}

const BENCHMARKS: ModelBenchmark[] = [
  {
    name: "XGBoost (Ours)",
    isOurs: true,
    rocAuc: "0.7909",
    prAuc: "0.2853",
    brier: "0.0654",
    ece: "0.0042",
    status: "Selected",
  },
  {
    name: "LightGBM",
    rocAuc: "0.7881",
    prAuc: "0.2819",
    brier: "0.0668",
    ece: "0.0056",
    status: "—",
  },
  {
    name: "Random Forest",
    rocAuc: "0.7796",
    prAuc: "0.2558",
    brier: "0.0712",
    ece: "0.0078",
    status: "—",
  },
  {
    name: "Logistic Regression",
    rocAuc: "0.7482",
    prAuc: "0.2337",
    brier: "0.0789",
    ece: "0.0104",
    status: "—",
  },
];

export function ModelComparisonCard({ rows = BENCHMARKS }: { rows?: ModelBenchmark[] }) {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <Layers className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">Model Comparison</h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Phase 11 out-of-sample benchmark evaluation on validation partition (N=46,127).
            </p>
          </div>
        </div>

        {/* Comparison Table */}
        <div className="overflow-x-auto mt-4">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#EEF2EE] text-[11.5px] font-semibold text-[#111613]">
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-2 text-center">ROC-AUC ↑</th>
                <th className="py-2.5 px-2 text-center">PR-AUC ↑</th>
                <th className="py-2.5 px-2 text-center">Brier ↓</th>
                <th className="py-2.5 px-2 text-center">ECE ↓</th>
                <th className="py-2.5 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="text-[12.5px]">
              {rows.map((row) => {
                const isSelected = row.status === "Selected";
                return (
                  <tr
                    key={row.name}
                    className={`transition-colors ${
                      isSelected
                        ? "bg-[#EAF5EE] font-semibold text-[#111613] rounded-xl"
                        : "text-[#1F2923] hover:bg-[#FAFBF9]"
                    }`}
                  >
                    {/* Model name */}
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        {row.isOurs && (
                          <div className="w-4 h-4 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0">
                            <Check className="w-2.5 h-2.5 stroke-[3]" />
                          </div>
                        )}
                        <span className={isSelected ? "font-bold text-[#111613]" : "font-medium"}>
                          {row.name}
                        </span>
                      </div>
                    </td>

                    {/* Metrics */}
                    <td className="py-3 px-2 text-center tabular-nums">
                      <span className={isSelected ? "font-bold text-[#111613]" : ""}>
                        {row.rocAuc}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-center tabular-nums">
                      <span className={isSelected ? "font-bold text-[#111613]" : ""}>
                        {row.prAuc}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-center tabular-nums">
                      <span className={isSelected ? "font-bold text-[#111613]" : ""}>
                        {row.brier}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-center tabular-nums">
                      <span className={isSelected ? "font-bold text-[#111613]" : ""}>
                        {row.ece}
                      </span>
                    </td>

                    {/* Status badge */}
                    <td className="py-3 px-3 text-center">
                      {isSelected ? (
                        <span className="inline-flex items-center justify-center px-3 py-0.5 rounded-full bg-[#158444] text-white text-[11px] font-semibold shadow-2xs">
                          Selected
                        </span>
                      ) : (
                        <span className="text-[#89938D] font-medium">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </article>
  );
}
