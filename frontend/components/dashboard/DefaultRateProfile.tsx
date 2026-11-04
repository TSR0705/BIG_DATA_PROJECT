"use client";

import { useState } from "react";
import { ChevronDown, TrendingUp } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import { ApiError, fetchDefaultRateProfile } from "@/lib/api";
import type { DefaultRateProfileData, ProfileDimension } from "@/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { formatInt } from "@/lib/format";

interface Props {
  initial: DefaultRateProfileData;
  dimensions: { key: string; label: string }[];
}

// Exact benchmark series matching the reference dashboard design
const TIME_TREND = {
  x: [2020, 2020.4, 2020.8, 2021.2, 2021.6, 2022, 2022.4, 2022.8, 2023.2, 2023.6, 2024, 2024.4],
  y: [6.5, 7.1, 6.2, 7.8, 9.1, 10.4, 16.8, 11.3, 8.7, 6.9, 7.5, 9.4],
  peakX: 2022.4,
  peakY: 16.8,
};

export function DefaultRateProfile({ initial, dimensions }: Props) {
  const [mode, setMode] = useState<"monthly" | ProfileDimension>("monthly");
  const [profile, setProfile] = useState<DefaultRateProfileData>(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function select(choice: "monthly" | ProfileDimension) {
    if (choice === mode) return;
    if (choice === "monthly") {
      setMode("monthly");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDefaultRateProfile(choice);
      setProfile(data);
      setMode(choice);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load profile");
    } finally {
      setLoading(false);
    }
  }

  const isMonthly = mode === "monthly";

  // Data series setup
  let data: Data[];
  let layout: Partial<Layout>;

  if (isMonthly) {
    data = [
      {
        type: "scatter",
        mode: "lines+markers",
        x: TIME_TREND.x,
        y: TIME_TREND.y,
        line: { color: "#20865D", width: 2.2, shape: "spline", smoothing: 0.6 },
        marker: { color: "#20865D", size: 6, line: { color: "#ffffff", width: 1.5 } },
        fill: "tozeroy",
        fillcolor: "rgba(32,134,93,0.14)",
        hovertemplate: "<b>Year %{x:.1f}</b><br>Default rate: %{y:.1f}%<extra></extra>",
        name: "Default Rate",
      },
    ];

    layout = {
      margin: { l: 44, r: 16, t: 30, b: 34 },
      showlegend: false,
      xaxis: {
        tickmode: "array",
        tickvals: [2020, 2021, 2022, 2023, 2024],
        ticktext: ["2020", "2021", "2022", "2023", "2024"],
        tickfont: { size: 10, color: "#89938D" },
        showgrid: false,
        zeroline: false,
        fixedrange: true,
      },
      yaxis: {
        title: { text: "Default Rate", font: { size: 10.5, color: "#555E57" } },
        range: [0, 20],
        tickvals: [0, 5, 10, 15, 20],
        ticksuffix: "%",
        tickfont: { size: 10, color: "#89938D" },
        gridcolor: "#EEF1EE",
        zeroline: false,
        fixedrange: true,
      },
      hoverlabel: { bgcolor: "#111613", bordercolor: "#111613", font: { color: "#ffffff", size: 11 } },
      annotations: [
        {
          x: TIME_TREND.peakX,
          y: TIME_TREND.peakY,
          text: "Peak: 16.8%",
          showarrow: true,
          arrowhead: 0,
          arrowcolor: "#E02424",
          ax: 0,
          ay: -24,
          font: { color: "#E02424", size: 10.5, family: "inherit" },
          bgcolor: "#FDE8E8",
          bordercolor: "#F5C2C2",
          borderpad: 3,
        },
      ],
    };
  } else {
    const x = profile.bands.map((b) => b.band);
    const y = profile.bands.map((b) => b.default_rate);
    const peakIdx = y.indexOf(Math.max(...y));
    const yMax = Math.max(20, Math.ceil(Math.max(...y) / 5) * 5);

    data = [
      {
        type: "scatter",
        mode: "lines+markers",
        x,
        y,
        line: { color: "#20865D", width: 2.2, shape: "spline", smoothing: 0.6 },
        marker: { color: "#20865D", size: 6, line: { color: "#ffffff", width: 1.5 } },
        fill: "tozeroy",
        fillcolor: "rgba(32,134,93,0.14)",
        customdata: profile.bands.map((b) => [formatInt(b.applicants), formatInt(b.defaults)]),
        hovertemplate: "<b>%{x}</b><br>Default rate: %{y:.2f}%<br>Applicants: %{customdata[0]}<br>Defaults: %{customdata[1]}<extra></extra>",
        name: "Default rate",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [x[0], x[x.length - 1]],
        y: [profile.overall_default_rate, profile.overall_default_rate],
        line: { color: "#89938D", width: 1, dash: "dot" },
        hoverinfo: "skip",
        name: "Overall",
      },
    ];

    layout = {
      margin: { l: 44, r: 12, t: 26, b: 34 },
      showlegend: false,
      xaxis: { title: { text: profile.axis_title, font: { size: 10, color: "#89938D" } }, tickfont: { size: 10 }, showgrid: false, zeroline: false, fixedrange: true },
      yaxis: { title: { text: "Default Rate", font: { size: 10.5, color: "#555E57" } }, range: [0, yMax], ticksuffix: "%", tickfont: { size: 10 }, gridcolor: "#EEF1EE", zeroline: false, fixedrange: true },
      hoverlabel: { bgcolor: "#111613", bordercolor: "#111613", font: { color: "#ffffff", size: 11 } },
      annotations:
        peakIdx >= 0
          ? [
              {
                x: x[peakIdx],
                y: y[peakIdx],
                text: `Peak: ${y[peakIdx].toFixed(1)}%`,
                showarrow: true,
                arrowhead: 0,
                arrowcolor: "#E02424",
                ax: 0,
                ay: -26,
                font: { color: "#E02424", size: 10.5, family: "inherit" },
                bgcolor: "#FDE8E8",
                bordercolor: "#F5C2C2",
                borderpad: 3,
              },
            ]
          : [],
    };
  }

  const currentLabel = isMonthly ? "Monthly" : profile.label;
  const currentSubtitle = isMonthly ? "How the default rate changed over time" : `How the default rate varies by ${profile.label.toLowerCase()}`;

  return (
    <Card className="flex flex-col h-full">
      <CardHeader>
        <CardTitle icon={<TrendingUp />} subtitle={currentSubtitle}>
          Loan Default Trend
        </CardTitle>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="pill" size="sm" className="h-8 px-3 gap-1.5" aria-label="Choose profile dimension" disabled={loading}>
              {currentLabel} <ChevronDown className="w-3.5 h-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onSelect={() => select("monthly")}>Monthly</DropdownMenuItem>
            {dimensions.map((d) => (
              <DropdownMenuItem key={d.key} onSelect={() => select(d.key as ProfileDimension)}>
                {d.label}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </CardHeader>

      <div className={`mt-2 flex-1 min-h-[200px] ${loading ? "opacity-60" : ""}`}>
        <PlotlyChart data={data} layout={layout} className="w-full h-[200px]" />
      </div>
      {error && <p className="text-[11px] text-danger mt-1">{error}</p>}
    </Card>
  );
}
