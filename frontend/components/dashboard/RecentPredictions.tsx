"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, Clock, UserRound } from "lucide-react";
import type { RecentPredictionData } from "@/types";
import { timeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

export function RecentPredictions({ predictions }: { predictions: RecentPredictionData[] }) {
  return (
    <div className="cl-card bg-white border border-[#E7ECE9] rounded-2xl p-4 sm:p-5 shadow-2xs flex flex-col flex-1">
      {/* Header: Inline clock icon + title + View All pill */}
      <div className="flex items-center justify-between pb-3 border-b border-[#F2F4F7]">
        <div className="flex items-center gap-2 min-w-0">
          <Clock className="w-4 h-4 text-[#087A43] stroke-[2.2] shrink-0" />
          <h3 className="text-[14px] font-bold text-[#101828] whitespace-nowrap truncate">
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
            className="mt-3 inline-flex items-center gap-1.5 text-[11px] font-semibold text-white bg-[#087A43] hover:bg-[#066336] px-3 py-1 rounded-full shadow-2xs transition-colors"
          >
            <span>Check Loan Risk</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      ) : (
        <ul className="divide-y divide-[#F2F4F7] flex-1 flex flex-col justify-between pt-1">
          {predictions.slice(0, 5).map((p) => {
            const isGenuine =
              p.inference_type === "GENUINE_MODEL" ||
              p.source === "search" ||
              p.sk_id_curr !== null;

            return (
              <li
                key={`${p.applicant_id}-${p.scored_at}`}
                className="flex items-center justify-between gap-2 py-2.5 hover:bg-[#F9FAF8] transition-colors rounded-lg px-1.5 -mx-1.5"
              >
                {/* Left: Avatar + Details */}
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className="w-7 h-7 rounded-full bg-[#F4F5F6] flex items-center justify-center text-[#555E57] shrink-0">
                    <UserRound className="w-3.5 h-3.5 stroke-[2]" />
                  </span>

                  <div className="min-w-0 leading-tight">
                    <Link
                      href={`/check-risk?id=${p.sk_id_curr ?? ""}`}
                      className="text-[11.5px] font-bold text-[#101828] hover:text-[#087A43] whitespace-nowrap block truncate transition-colors"
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
                <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
                  <span
                    className={cn(
                      "text-[9.5px] sm:text-[10px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap text-center",
                      p.risk_tier === "Low Risk" && "bg-[#E0F9E6] text-[#087A43]",
                      p.risk_tier === "Medium Risk" && "bg-[#FEF6EE] text-[#B54708]",
                      p.risk_tier === "High Risk" && "bg-[#FDE8E8] text-[#E02424]",
                    )}
                  >
                    {p.risk_tier}
                  </span>
                  <span className="text-[11.5px] sm:text-[12px] font-bold text-[#101828] tabular-nums text-right whitespace-nowrap w-10 sm:w-11">
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