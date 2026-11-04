import { AlertTriangle, ArrowUpRight, Target, Users, type LucideIcon } from "lucide-react";
import type { KPICardData } from "@/types";
import { Sparkline } from "./Sparkline";

const STYLE: Record<string, { icon: LucideIcon; iconBg: string; stroke: string; sub: string }> = {
  green: { icon: Users, iconBg: "bg-[#E0F9E6] text-[#158444]", stroke: "#158444", sub: "text-[#158444]" },
  red: { icon: AlertTriangle, iconBg: "bg-[#FDE8E8] text-[#E02424]", stroke: "#F87171", sub: "text-[#E02424]" },
  teal: { icon: Target, iconBg: "bg-[#E1F7F5] text-[#088395]", stroke: "#158444", sub: "text-[#555E57]" },
};

export function KPICards({ kpis }: { kpis: KPICardData[] }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 h-full">
      {kpis.map((kpi) => {
        const s = STYLE[kpi.badge_type] ?? STYLE.green;
        const Icon = s.icon;
        const hasArrow = kpi.trend === "up" && kpi.title === "Total Applicants";
        return (
          <article key={kpi.title} className="cl-card p-5 flex flex-col justify-between" title={`Source: ${kpi.source}`}>
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${s.iconBg}`}>
                <Icon className="w-5 h-5 stroke-[2.2]" />
              </div>
              <span className="text-[13px] font-semibold text-ink-muted">{kpi.title}</span>
            </div>
            <div className="my-2.5">
              <span className="text-3xl font-extrabold text-ink tracking-tight tabular-nums">{kpi.value}</span>
            </div>
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-1 min-w-0">
                {hasArrow && <ArrowUpRight className={`w-3.5 h-3.5 ${s.sub} stroke-[2.5] shrink-0`} />}
                <span className={`text-[12px] font-semibold ${s.sub} truncate`}>{kpi.subtitle}</span>
              </div>
              <Sparkline values={kpi.sparkline} stroke={s.stroke} label={kpi.sparkline_label} />
            </div>
          </article>
        );
      })}
    </div>
  );
}
