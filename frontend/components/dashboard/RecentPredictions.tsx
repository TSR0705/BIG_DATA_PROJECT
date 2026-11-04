"use client";

import Link from "next/link";
import { Clock, UserRound } from "lucide-react";
import type { RecentPredictionData } from "@/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { timeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

export function RecentPredictions({ predictions }: { predictions: RecentPredictionData[] }) {
  return (
    <Card className="flex flex-col flex-1">
      <CardHeader className="items-center pb-2">
        <CardTitle icon={<Clock />}>Recent Predictions</CardTitle>
        <Link
          href="/understand-results"
          className="inline-flex items-center justify-center bg-white border border-[#D1D5DB] hover:bg-neutral-50 text-ink text-[11px] font-semibold px-3 py-1 rounded-full transition-colors shadow-2xs"
        >
          View All
        </Link>
      </CardHeader>
      {predictions.length === 0 ? (
        <p className="text-xs text-ink-muted mt-4 px-5">No predictions have been made yet.</p>
      ) : (
        <ul className="divide-y divide-app/80 mt-1 px-5">
          {predictions.map((p) => (
            <li key={`${p.applicant_id}-${p.scored_at}`} className="flex items-center justify-between gap-2 py-2.5">
              <div className="flex items-center gap-2.5 min-w-0">
                <span className="w-8 h-8 rounded-full bg-[#F4F5F6] flex items-center justify-center text-[#555E57] shrink-0">
                  <UserRound className="w-4 h-4 stroke-[2]" />
                </span>
                <div className="min-w-0 leading-tight">
                  <Link href={`/check-risk?id=${p.sk_id_curr ?? ""}`} className="text-[12px] font-bold text-ink hover:text-[#158444] truncate block">
                    {p.applicant_id}
                  </Link>
                  <span className="text-[10px] text-[#89938D]">
                    {timeAgo(p.scored_at)}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span
                  className={cn(
                    "text-[10.5px] font-semibold px-2 py-0.5 rounded-full",
                    p.risk_tier === "Low Risk" && "bg-[#E0F9E6] text-[#158444]",
                    p.risk_tier === "Medium Risk" && "bg-[#FEF9C3] text-[#A16207]",
                    p.risk_tier === "High Risk" && "bg-[#FDE8E8] text-[#E02424]",
                  )}
                >
                  {p.risk_tier}
                </span>
                <span className="text-[12px] font-bold text-ink tabular-nums w-10 text-right">{p.probability_percent}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
