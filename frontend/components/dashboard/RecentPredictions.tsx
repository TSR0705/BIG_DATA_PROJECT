"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, Clock, UserRound } from "lucide-react";
import type { RecentPredictionData } from "@/types";
import { timeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

export function RecentPredictions({ predictions }: { predictions: RecentPredictionData[] }) {
  return (
    <div className="cl-card bg-white border border-[#E8EDE9] rounded-2xl p-4 sm:p-5 shadow-2xs flex flex-col flex-1 justify-between">
      {/* Header: Clock icon badge + title + View All pill */}
      <div className="flex items-center justify-between pb-3 border-b border-[#F2F4F7]">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-[#E0F9E6] text-[#087A43] flex items-center justify-center shrink-0">
            <Clock className="w-3.5 h-3.5 stroke-[2.4]" />
          </div>
          <h3 className="text-[14.5px] font-bold text-[#101828] whitespace-nowrap truncate tracking-tight">
            Recent Predictions
          </h3>
        </div>
        <Link
          href="/understand-results"
          className="inline-flex items-center justify-center bg-white border border-[#D0D5DD] hover:bg-[#F9FAFB] hover:border-[#98A2B3] text-[#344054] text-[11px] font-semibold px-2.5 py-0.5 rounded-full transition-colors whitespace-nowrap shrink-0 shadow-2xs"
          title="Explore full explainability and prediction history"
        >
          View All
        </Link>
      </div>

      {predictions.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-5 text-center my-auto min-h-[180px]">
          <div className="w-9 h-9 rounded-full bg-[#EBF7EF] flex items-center justify-center text-[#087A43] mb-2.5">
            <Clock className="w-4 h-4 stroke-[2]" />
          </div>
          <p className="text-[12.5px] font-bold text-[#101828]">No recent predictions</p>
          <p className="text-[11px] text-[#667085] mt-1 max-w-[190px] leading-relaxed">
            Evaluate loan risk for an applicant to view live scoring history here.
          </p>
          <Link
            href="/check-risk"
            className="mt-3 inline-flex items-center gap-1.5 text-[11px] font-semibold text-white bg-[#087A43] hover:bg-[#066336] px-3.5 py-1 rounded-full shadow-2xs transition-colors"
          >
            <span>Check Loan Risk</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      ) : (
        <ul className="divide-y divide-[#F4F6F4] flex-1 flex flex-col justify-between pt-1">
          {predictions.slice(0, 5).map((p) => {
            const isGenuine =
              p.inference_type === "GENUINE_MODEL" ||
              p.source === "search" ||
              p.sk_id_curr !== null;

            const isLow = p.risk_tier.toLowerCase().includes("low");
            const isMedium = p.risk_tier.toLowerCase().includes("med");
            const isHigh = p.risk_tier.toLowerCase().includes("high");

            return (
              <li
                key={`${p.applicant_id}-${p.scored_at}`}
                className="flex items-center justify-between gap-2 py-2.5 px-2 hover:bg-[#F8FAF8] transition-colors rounded-xl -mx-1"
              >
                {/* Left: Avatar + Details */}
                <div className="flex items-center gap-2.5 min-w-0 flex-1">
                  <span
                    className={cn(
                      "w-7 h-7 rounded-full flex items-center justify-center shrink-0 shadow-2xs",
                      isLow && "bg-[#EAF5EE] text-[#087A43]",
                      isMedium && "bg-[#FEF6EE] text-[#B54708]",
                      isHigh && "bg-[#FDE8E8] text-[#E02424]",
                      !isLow && !isMedium && !isHigh && "bg-[#F4F5F6] text-[#555E57]",
                    )}
                  >
                    <UserRound className="w-3.5 h-3.5 stroke-[2.2]" />
                  </span>

                  <div className="min-w-0 leading-tight">
                    <Link
                      href={`/check-risk?id=${p.sk_id_curr ?? ""}`}
                      className="text-[12px] font-bold text-[#101828] hover:text-[#087A43] whitespace-nowrap block truncate transition-colors"
                      title={
                        isGenuine
                          ? `Applicant ${p.applicant_id} — Genuine 256-Feature XGBoost Model`
                          : `Applicant ${p.applicant_id} — Illustrative Heuristic Estimate`
                      }
                    >
                      {p.applicant_id}
                    </Link>
                    <span className="text-[10px] text-[#89938D] whitespace-nowrap block truncate mt-0.5">
                      {timeAgo(p.scored_at)}
                    </span>
                  </div>
                </div>

                {/* Right: Badge + Probability */}
                <div className="flex items-center gap-2 shrink-0">
                  <span
                    className={cn(
                      "text-[10px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap text-center inline-flex items-center gap-1",
                      isLow && "bg-[#E0F9E6] text-[#087A43] border border-[#A6F4C5]/60",
                      isMedium && "bg-[#FEF3C7] text-[#B54708] border border-[#FDE68A]/60",
                      isHigh && "bg-[#FEE2E2] text-[#DC2626] border border-[#FECDCA]/60",
                    )}
                  >
                    <span
                      className={cn(
                        "w-1.5 h-1.5 rounded-full shrink-0",
                        isLow && "bg-[#087A43]",
                        isMedium && "bg-[#B54708]",
                        isHigh && "bg-[#DC2626]",
                      )}
                    />
                    {p.risk_tier}
                  </span>
                  <span className="text-[12px] font-extrabold text-[#101828] tabular-nums text-right whitespace-nowrap w-11">
                    {p.probability_percent}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
