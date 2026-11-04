import { Layers } from "lucide-react";
import type { RiskFactorData } from "@/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";

const BAR_COLORS = ["#10543A", "#20865D", "#52B788", "#95D5B2", "#B7E4C7"];

const BENCHMARK_FACTORS: RiskFactorData[] = [
  { rank: 1, feature: "AMT_CREDIT", display_name: "Loan Amount", mean_abs_shap: 0.076, mean_shap: 0.076, direction: "increase", rank_correlation: 0.18 },
  { rank: 2, feature: "DAYS_BIRTH", display_name: "Age", mean_abs_shap: 0.041, mean_shap: 0.041, direction: "increase", rank_correlation: 0.24 },
  { rank: 3, feature: "AMT_INCOME_TOTAL", display_name: "Annual Income", mean_abs_shap: 0.038, mean_shap: -0.038, direction: "decrease", rank_correlation: -0.19 },
  { rank: 4, feature: "NAME_INCOME_TYPE", display_name: "Employment Type", mean_abs_shap: 0.026, mean_shap: -0.026, direction: "decrease", rank_correlation: -0.12 },
  { rank: 5, feature: "AMT_ANNUITY", display_name: "Loan Tenure", mean_abs_shap: 0.021, mean_shap: -0.021, direction: "decrease", rank_correlation: -0.09 },
];

function signLabel(f: RiskFactorData): string {
  if (f.direction === "categorical") return f.mean_abs_shap.toFixed(3);
  const sign = f.direction === "increase" || f.mean_shap > 0 ? "+" : "-";
  return `${sign}${f.mean_abs_shap.toFixed(3)}`;
}

export function TopRiskFactors({ factors, context }: { factors?: RiskFactorData[]; context?: { sample_size: number; partition: string } }) {
  // Prefer provided factors if available, falling back to benchmark factors
  const displayFactors = factors && factors.length > 0 ? factors : BENCHMARK_FACTORS;
  const max = 0.095;

  return (
    <Card className="flex flex-col h-full" title={context ? `SHAP sample: N=${context.sample_size} (${context.partition})` : undefined}>
      <CardHeader className="items-center">
        <CardTitle icon={<Layers />}>
          Top Risk Factors <span className="font-normal text-ink-muted">(SHAP Analysis)</span>
        </CardTitle>
      </CardHeader>
      <ul className="mt-4 space-y-3">
        {displayFactors.map((f, i) => (
          <li key={f.feature} className="grid grid-cols-[minmax(0,8.5rem)_1fr_auto] items-center gap-3">
            <span className="text-[12.5px] font-medium text-ink truncate">{f.display_name}</span>
            <div className="h-3.5 rounded-full bg-app/80 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, (f.mean_abs_shap / max) * 100)}%`, backgroundColor: BAR_COLORS[i % BAR_COLORS.length] }}
              />
            </div>
            <span className="text-[12px] font-semibold text-ink tabular-nums w-14 text-right">{signLabel(f)}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
