"use client";

import { UserRound } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import type { ApplicantDistributionData } from "@/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { formatInt } from "@/lib/format";

export function ApplicantDistribution({ distribution }: { distribution: ApplicantDistributionData }) {
  const data: Data[] = [
    {
      type: "pie",
      hole: 0.62,
      values: [distribution.non_default_count, distribution.default_count],
      labels: ["Non-Default", "Default"],
      marker: { colors: ["#10543A", "#48C774"], line: { color: "#ffffff", width: 2.5 } },
      textinfo: "none",
      sort: false,
      direction: "clockwise",
      rotation: 90,
      hovertemplate: "<b>%{label}</b><br>%{value:,} applicants (%{percent})<extra></extra>",
    },
  ];
  const layout: Partial<Layout> = {
    margin: { l: 4, r: 4, t: 4, b: 4 },
    showlegend: false,
    hoverlabel: { bgcolor: "#111613", bordercolor: "#111613", font: { color: "#ffffff", size: 11 } },
    annotations: [
      { text: `<b>${formatInt(distribution.total)}</b>`, x: 0.5, y: 0.56, showarrow: false, font: { size: 18, color: "#111613" } },
      { text: "Applicants", x: 0.5, y: 0.4, showarrow: false, font: { size: 11, color: "#555E57" } },
    ],
  };

  return (
    <Card className="flex flex-col h-full">
      <CardHeader>
        <CardTitle icon={<UserRound />} iconClass="bg-[#E0F9E6] text-[#10543A]">
          Applicant Distribution
        </CardTitle>
      </CardHeader>
      <div className="flex-1 flex flex-col sm:flex-row items-center gap-4 mt-2">
        <PlotlyChart data={data} layout={layout} className="w-[160px] h-[160px] shrink-0" />
        <ul className="space-y-3.5 text-[12.5px]">
          <li className="flex items-start gap-2.5">
            <span className="w-3.5 h-3.5 rounded-full bg-[#10543A] mt-0.5 shrink-0 shadow-2xs" />
            <span>
              <span className="font-bold text-[#111613]">Non-Default</span>
              <br />
              <span className="text-[#555E57] font-medium tabular-nums">
                {formatInt(distribution.non_default_count)} ({distribution.non_default_percent.toFixed(1)}%)
              </span>
            </span>
          </li>
          <li className="flex items-start gap-2.5">
            <span className="w-3.5 h-3.5 rounded-full bg-[#48C774] mt-0.5 shrink-0 shadow-2xs" />
            <span>
              <span className="font-bold text-[#111613]">Default</span>
              <br />
              <span className="text-[#555E57] font-medium tabular-nums">
                {formatInt(distribution.default_count)} ({distribution.default_percent.toFixed(1)}%)
              </span>
            </span>
          </li>
        </ul>
      </div>
    </Card>
  );
}
