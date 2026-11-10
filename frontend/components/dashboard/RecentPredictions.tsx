"use client";

import React, { useState } from "react";
import Link from "next/link";
import { ArrowRight, Check, Clock, Copy, UserRound } from "lucide-react";
import type { RecentPredictionData } from "@/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { timeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

export function RecentPredictions({ predictions }: { predictions: RecentPredictionData[] }) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(id).catch(() => {});
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1800);
    }
  };

  return (
    <Card className="flex flex-col flex-1 bg-white border border-[#E7ECE9] rounded-2xl shadow-2xs overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between pb-2.5 pt-4 px-5 border-b border-[#F2F4F7]">
        <CardTitle
          icon={<Clock className="w-4 h-4 text-[#087A43] stroke-[2.2]" />}
          className="text-[14.5px] sm:text-[15px] font-bold text-[#101828]"
        >
          Recent Predictions
        </CardTitle>
        <Link
          href="/understand-results"
          className="inline-flex items-center justify-center bg-white border border-[#D0D5DD] hover:bg-[#F9FAFB] hover:border-[#98A2B3] text-[#344054] text-[11px] font-semibold px-3 py-1 rounded-full transition-all shadow-2xs"
          title="Explore full explainability and prediction history"
        >
          View All
        </Link>
      </CardHeader>

      {predictions.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-6 text-center my-auto min-h-[220px]">
          <div className="w-10 h-10 rounded-full bg-[#EBF7EF] flex items-center justify-center text-[#087A43] mb-3">
            <Clock className="w-5 h-5 stroke-[2]" />
          </div>
          <p className="text-[13px] font-bold text-[#101828]">No recent predictions</p>
          <p className="text-[11.5px] text-[#667085] mt-1 max-w-[210px] leading-relaxed">
            Evaluate loan risk for an applicant to view live scoring history here.
          </p>
          <Link
            href="/check-risk"
            className="mt-3.5 inline-flex items-center gap-1.5 text-[11.5px] font-semibold text-white bg-[#087A43] hover:bg-[#066336] px-3.5 py-1.5 rounded-full shadow-2xs transition-colors"
          >
            <span>Check Loan Risk</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      ) : (
        <ul className="divide-y divide-[#F2F4F7] py-1 px-4 sm:px-5">
          {predictions.map((p) => {
            const isGenuine =
              p.inference_type === "GENUINE_MODEL" ||
              p.source === "search" ||
              p.sk_id_curr !== null;
            const isCopied = copiedId === p.applicant_id;

            return (
              <li
                key={`${p.applicant_id}-${p.scored_at}`}
                className="group flex items-center justify-between gap-2.5 py-2.5 min-h-[50px] hover:bg-[#F8FAF9] transition-colors rounded-xl px-2 -mx-2"
              >
                {/* Left: Avatar + ID & Provenance */}
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="w-8 h-8 rounded-full bg-[#F2F4F7] group-hover:bg-[#EBF7EF] flex items-center justify-center text-[#667085] group-hover:text-[#087A43] shrink-0 transition-colors">
                    <UserRound className="w-4 h-4 stroke-[2]" />
                  </span>

                  <div className="min-w-0 leading-tight">
                    <div className="flex items-center gap-1.5">
                      <Link
                        href={`/check-risk?id=${p.sk_id_curr ?? ""}`}
                        className="text-[12px] sm:text-[12.5px] font-bold text-[#101828] group-hover:text-[#087A43] whitespace-nowrap block truncate transition-colors"
                        title={
                          p.sk_id_curr
                            ? `Inspect Applicant SK_ID_CURR ${p.sk_id_curr}`
                            : `Custom applicant ${p.applicant_id}`
                        }
                      >
                        {p.applicant_id}
                      </Link>

                      <button
                        type="button"
                        onClick={(e) => handleCopy(p.applicant_id, e)}
                        className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-0.5 text-[#98A2B3] hover:text-[#087A43] transition-all rounded hover:bg-[#EBF7EF]"
                        title={isCopied ? "Copied to clipboard!" : "Copy Applicant ID"}
                        aria-label={`Copy ID ${p.applicant_id}`}
                      >
                        {isCopied ? (
                          <Check className="w-3 h-3 text-[#087A43] stroke-[2.5]" />
                        ) : (
                          <Copy className="w-3 h-3 stroke-[2]" />
                        )}
                      </button>
                    </div>

                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className="text-[10.5px] sm:text-[11px] text-[#667085] whitespace-nowrap">
                        {timeAgo(p.scored_at)}
                      </span>
                      <span className="text-[#D0D5DD] text-[10px]">·</span>
                      {isGenuine ? (
                        <span
                          className="inline-flex items-center text-[9.5px] font-semibold text-[#087A43] bg-[#EBF7EF] px-1.5 py-0.2 rounded tracking-tight"
                          title="Evaluated with genuine 256-feature XGBoost model + Isotonic Calibrator"
                        >
                          XGBoost
                        </span>
                      ) : (
                        <span
                          className="inline-flex items-center text-[9.5px] font-medium text-[#64748B] bg-[#F1F5F9] px-1.5 py-0.2 rounded tracking-tight"
                          title="Illustrative Heuristic Estimate (6 form inputs)"
                        >
                          Heuristic
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right: Risk Badge + Probability */}
                <div className="flex items-center gap-2.5 shrink-0">
                  <span
                    className={cn(
                      "text-[10.5px] sm:text-[11px] font-semibold px-2.5 py-0.5 rounded-full min-w-[76px] sm:min-w-[80px] text-center border transition-colors",
                      p.risk_tier === "Low Risk" && "bg-[#ECFDF3] text-[#027A48] border-[#A6F4C5]/80",
                      p.risk_tier === "Medium Risk" && "bg-[#FEF6EE] text-[#B54708] border-[#FECDCA]/60",
                      p.risk_tier === "High Risk" && "bg-[#FEF3F2] text-[#B42318] border-[#FECDCA]/80",
                    )}
                  >
                    {p.risk_tier}
                  </span>
                  <span className="text-[12px] sm:text-[12.5px] font-extrabold text-[#101828] tabular-nums w-12 text-right tracking-tight">
                    {p.probability_percent}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}