import { AlertTriangle, ArrowUpRight, Target, Users, type LucideIcon } from "lucide-react";
import type { KPICardData } from "@/types";
import { Sparkline } from "./Sparkline";

interface StyleDef {
  icon: LucideIcon;
  iconBg: string;
  stroke: string;
  sub: string;
  badgeBg: string;
}

const STYLE: Record<string, StyleDef> = {
  green: {
    icon: Users,
    iconBg: "bg-[#E0F9E6] text-[#158444]",
    stroke: "#158444",
    sub: "text-[#158444]",
    badgeBg: "bg-[#EAF5EE]",
  },
  red: {
    icon: AlertTriangle,
    iconBg: "bg-[#FEE2E2] text-[#DC2626]",
    stroke: "#F87171",
    sub: "text-[#DC2626]",
    badgeBg: "bg-[#FEF2F2]",
  },
  teal: {
    icon: Target,
    iconBg: "bg-[#E0F2FE] text-[#0284C7]",
    stroke: "#0284C7",
    sub: "text-[#555E57]",
    badgeBg: "bg-[#F0F9FF]",
  },
};

export function KPICards({ kpis }: { kpis: KPICardData[] }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 h-full">
      {kpis.map((kpi) => {
        const s = STYLE[kpi.badge_type] ?? STYLE.green;
        const Icon = s.icon;
        const hasArrow = kpi.trend === "up" && kpi.title === "Total Applicants";
        return (
          <article
            key={kpi.title}
            className="cl-card p-5 sm:p-5.5 flex flex-col justify-between hover:border-[#D5DDD6] transition-all"
            title={`Source: ${kpi.source}`}
          >
            {/* Top row: Icon Badge + Label */}
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${s.iconBg} shadow-2xs`}>
                <Icon className="w-5 h-5 stroke-[2.2]" />
              </div>
              <span className="text-[13px] font-semibold text-[#555E57] tracking-tight">{kpi.title}</span>
            </div>

            {/* Middle: Big Stat */}
            <div className="my-2.5">
              <span className="text-[28px] sm:text-[30px] font-extrabold text-[#111613] tracking-tight tabular-nums leading-none">
                {kpi.value}
              </span>
            </div>

            {/* Bottom: Subtitle with Indicator + Sparkline */}
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-1.5 min-w-0">
                {hasArrow ? (
                  <span className="inline-flex items-center gap-1 text-[11.5px] font-bold text-[#158444]">
                    <ArrowUpRight className="w-3.5 h-3.5 stroke-[2.6] shrink-0" />
                    <span className="truncate">{kpi.subtitle}</span>
                  </span>
                ) : (
                  <span className={`text-[11.5px] font-semibold ${s.sub} truncate`}>
                    {kpi.subtitle}
                  </span>
                )}
              </div>
              <Sparkline values={kpi.sparkline} stroke={s.stroke} label={kpi.sparkline_label} />
            </div>
          </article>
        );
      })}
    </div>
  );
}
