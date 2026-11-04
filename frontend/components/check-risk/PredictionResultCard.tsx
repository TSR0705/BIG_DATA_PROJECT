"use client";

import React from "react";
import { AlertTriangle, BarChart3, Check, GitFork, Layers, ShieldCheck } from "lucide-react";
import type { PredictRiskResult } from "@/types";
import { GaugeChart } from "./GaugeChart";
import { cn } from "@/lib/utils";

interface Props {
  result: PredictRiskResult;
}

export function PredictionResultCard({ result }: Props) {
  const isLow = result.risk_tier === "Low Risk";
  const isHigh = result.risk_tier === "High Risk";

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-3">
          <span className="w-10 h-10 rounded-xl bg-[#E0F9E6] flex items-center justify-center text-[#158444] shrink-0">
            <BarChart3 className="w-5 h-5 stroke-[2.2]" />
          </span>
          <h2 className="text-[17px] font-bold text-[#111613]">Prediction Result</h2>
        </div>

        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#EAF5EE] text-[#158444] text-[12px] font-semibold">
          <Check className="w-3.5 h-3.5 stroke-[2.8]" />
          <span>Prediction Completed</span>
        </div>
      </div>

      {/* Middle Section: Centered vertically in available card height */}
      <div className="flex-1 flex flex-col justify-center my-auto py-4">
        <div className="grid grid-cols-1 sm:grid-cols-12 items-center gap-4">
          {/* Gauge chart */}
          <div className="sm:col-span-6 flex justify-center items-center">
            <GaugeChart percentage={result.default_percentage} />
          </div>

          {/* Risk Badge and Explanation */}
          <div className="sm:col-span-6 flex flex-col items-start justify-center pl-1 sm:pl-3">
            <div
              className={cn(
                "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl font-bold text-[13px] shadow-2xs",
                isLow && "bg-[#E0F9E6] text-[#158444] border border-[#C6F2D2]",
                !isLow && !isHigh && "bg-[#FEF9C3] text-[#854D0E] border border-[#FEF08A]",
                isHigh && "bg-[#FDE8E8] text-[#E02424] border border-[#FCD2D2]",
              )}
            >
              <AlertTriangle className="w-4 h-4 stroke-[2.2]" />
              <span>{result.risk_tier}</span>
            </div>

            <p className="text-[13px] text-[#555E57] leading-relaxed mt-3 max-w-[270px]">
              {result.summary}
            </p>

            {result.inference_type === "ILLUSTRATIVE_HEURISTIC" && (
              <div className="mt-2.5 text-[11px] leading-relaxed text-[#854D0E] bg-[#FEFCE8] border border-[#FEF08A] rounded-xl p-2.5 max-w-[275px]">
                <span className="font-bold">⚠️ Demonstration Only:</span> Illustrative heuristic estimate based on 6 core inputs. Not the research 256-feature calibrated XGBoost model and not a validated lending probability.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom 3 Summary Tiles: No divider line, proportional widths so XGBoost (Calibrated) never truncates */}
      <div className="flex flex-col sm:flex-row items-stretch gap-2.5 shrink-0 pt-1">
        {/* Risk Category */}
        <div className="flex-[1] min-w-0 bg-[#F8FAF8] border border-[#EEF2EE] rounded-2xl p-3 flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0">
            <Layers className="w-4 h-4 stroke-[2]" />
          </span>
          <div className="min-w-0">
            <p className="text-[10.5px] text-[#89938D] font-medium leading-none">Risk Category</p>
            <p className="text-[13.5px] font-bold text-[#111613] mt-1 truncate">{result.risk_category}</p>
          </div>
        </div>

        {/* Confidence */}
        <div className="flex-[1] min-w-0 bg-[#F8FAF8] border border-[#EEF2EE] rounded-2xl p-3 flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0">
            <ShieldCheck className="w-4 h-4 stroke-[2]" />
          </span>
          <div className="min-w-0">
            <p className="text-[10.5px] text-[#89938D] font-medium leading-none">Confidence</p>
            <p className="text-[13.5px] font-bold text-[#111613] mt-1 truncate">{result.confidence}</p>
          </div>
        </div>

        {/* Model Used (flex-[1.35] gives ample room for XGBoost (Calibrated)) */}
        <div className="flex-[1.35] min-w-0 bg-[#F8FAF8] border border-[#EEF2EE] rounded-2xl p-3 flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0">
            <GitFork className="w-4 h-4 stroke-[2]" />
          </span>
          <div className="min-w-0">
            <p className="text-[10.5px] text-[#89938D] font-medium leading-none">Model Used</p>
            <p className="text-[12.5px] font-bold text-[#111613] mt-1 whitespace-nowrap">{result.model_used}</p>
          </div>
        </div>
      </div>
    </article>
  );
}
