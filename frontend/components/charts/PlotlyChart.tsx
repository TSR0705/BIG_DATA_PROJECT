"use client";

import dynamic from "next/dynamic";
import type { Config, Data, Layout } from "plotly.js";
import { Skeleton } from "@/components/ui/skeleton";

/** Plotly is browser-only; load it lazily with a skeleton placeholder. */
const Plot = dynamic(
  async () => {
    const [factoryModule, plotlyModule] = await Promise.all([import("react-plotly.js/factory"), import("plotly.js-dist-min")]);
    const createPlotlyComponent = factoryModule.default;
    const plotly = (plotlyModule as unknown as { default?: unknown }).default ?? plotlyModule;
    return createPlotlyComponent(plotly as never);
  },
  { ssr: false, loading: () => <Skeleton className="w-full h-full" /> },
);

export const PLOT_FONT = { family: 'ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', color: "#555E57", size: 11 };

export const BASE_CONFIG: Partial<Config> = { displayModeBar: false, responsive: true, staticPlot: false };

export function PlotlyChart({ data, layout, config, className }: { data: Data[]; layout: Partial<Layout>; config?: Partial<Config>; className?: string }) {
  return (
    <div className={className}>
      <Plot
        data={data}
        layout={{ autosize: true, font: PLOT_FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", ...layout }}
        config={{ ...BASE_CONFIG, ...config }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
