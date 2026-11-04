"use client";

import React from "react";
import { AlertTriangle, CircleDot, GitFork, Layers, ShieldCheck } from "lucide-react";
import { GaugeChart } from "@/components/check-risk/GaugeChart";
import { cn } from "@/lib/utils";

interface Props {
  applicantId?: string;
  percentage?: number;
  riskTier?: string;
  riskCategory?: string;
  confidence?: string;
  modelUsed?: string;
  summary?: string;
  inferenceType?: string;
  disclaimer?: string;
}

export function PredictionSummaryCard({
  applicantId = "APP-105781",
  percentage = 14.2,
  riskTier = "Medium Risk",
  riskCategory = "Medium",
  confidence = "Demonstration Only",
  modelUsed = "Illustrative Heuristic Estimate",
  summary = "Illustrative heuristic estimate: 14.2% likelihood based on 6 input ratios (demonstration only, not a validated credit risk assessment).",
  inferenceType = "ILLUSTRATIVE_HEURISTIC",
  disclaimer,
}: Props) {
  const isLow = riskTier === "Low Risk";
  const isHigh = riskTier === "High Risk";

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 shrink-0 mb-2">
        <div className="flex items-center gap-2.5">
          <CircleDot className="w-6 h-6 text-[#158444] stroke-[2.3] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Prediction Summary</h2>
        </div>

        <div className="inline-flex items-center px-3.5 py-1 rounded-full bg-[#F2F4F2] text-[#555E57] text-[11.5px] font-medium">
          Applicant ID: {applicantId}
        </div>
      </div>

      {/* Middle Section: Centered vertically */}
      <div className="flex-1 flex flex-col justify-center my-auto py-3">
        <div className="grid grid-cols-1 sm:grid-cols-12 items-center gap-4">
          {/* Gauge chart */}
          <div className="sm:col-span-6 flex justify-center items-center">
            <GaugeChart percentage={percentage} />
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
              <span>{riskTier}</span>
            </div>

            <p className="text-[13px] text-[#555E57] leading-relaxed mt-3 max-w-[270px]">
              {summary}
            </p>

            {inferenceType === "ILLUSTRATIVE_HEURISTIC" && (
              <div className="mt-2.5 text-[11px] leading-relaxed text-[#854D0E] bg-[#FEFCE8] border border-[#FEF08A] rounded-xl p-2.5 max-w-[275px]">
                <span className="font-bold">⚠️ Demonstration Only:</span>{" "}
                {disclaimer ||
                  "Illustrative heuristic estimate based on 6 core inputs. Not the research 256-feature calibrated XGBoost model and not a validated lending probability."}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom 3 Summary Tiles */}
      <div className="flex flex-col sm:flex-row items-stretch gap-2.5 shrink-0 pt-1">
        {/* Risk Category */}
        <div className="flex-[1] min-w-0 bg-[#F8FAF8] border border-[#EEF2EE] rounded-2xl p-3 flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0">
            <Layers className="w-4 h-4 stroke-[2]" />
          </span>
          <div className="min-w-0">
            <p className="text-[10.5px] text-[#89938D] font-medium leading-none">Risk Category</p>
            <p className="text-[13.5px] font-bold text-[#111613] mt-1 truncate">{riskCategory}</p>
          </div>
        </div>

        {/* Confidence */}
        <div className="flex-[1] min-w-0 bg-[#F8FAF8] border border-[#EEF2EE] rounded-2xl p-3 flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0">
            <ShieldCheck className="w-4 h-4 stroke-[2]" />
          </span>
          <div className="min-w-0">
            <p className="text-[10.5px] text-[#89938D] font-medium leading-none">Confidence</p>
            <p className="text-[13.5px] font-bold text-[#111613] mt-1 truncate">{confidence}</p>
          </div>
        </div>

        {/* Model Used */}
        <div className="flex-[1.35] min-w-0 bg-[#F8FAF8] border border-[#EEF2EE] rounded-2xl p-3 flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-full bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0">
            <GitFork className="w-4 h-4 stroke-[2]" />
          </span>
          <div className="min-w-0">
            <p className="text-[10.5px] text-[#89938D] font-medium leading-none">Model Used</p>
            <p className="text-[12.5px] font-bold text-[#111613] mt-1 whitespace-nowrap">{modelUsed}</p>
          </div>
        </div>
      </div>
    </article>
  );
}
