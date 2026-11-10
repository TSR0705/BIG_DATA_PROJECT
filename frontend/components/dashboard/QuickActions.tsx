import Link from "next/link";
import { ArrowRight, BarChart3, BookOpen, Database, UserPlus, Zap, type LucideIcon } from "lucide-react";

interface ActionItem {
  title: string;
  href: string;
  icon: LucideIcon;
  iconBg: string;
  hoverBg: string;
  hoverBorder: string;
  accent: string;
}

const ACTIONS: ActionItem[] = [
  {
    title: "Check Loan Risk",
    href: "/check-risk",
    icon: UserPlus,
    iconBg: "bg-[#E0F9E6] text-[#158444]",
    hoverBg: "hover:bg-[#F4FAF6]",
    hoverBorder: "hover:border-[#158444]/30",
    accent: "text-[#158444]",
  },
  {
    title: "View Model Performance",
    href: "/model-performance",
    icon: BarChart3,
    iconBg: "bg-[#E0F2FE] text-[#0284C7]",
    hoverBg: "hover:bg-[#F0F9FF]",
    hoverBorder: "hover:border-[#0284C7]/30",
    accent: "text-[#0284C7]",
  },
  {
    title: "Explore Dataset",
    href: "/data-pipeline",
    icon: Database,
    iconBg: "bg-[#EAF3EE] text-[#10543A]",
    hoverBg: "hover:bg-[#F4F8F5]",
    hoverBorder: "hover:border-[#10543A]/30",
    accent: "text-[#10543A]",
  },
  {
    title: "See Research",
    href: "/research-insights",
    icon: BookOpen,
    iconBg: "bg-[#FEF3C7] text-[#D97706]",
    hoverBg: "hover:bg-[#FFFDF5]",
    hoverBorder: "hover:border-[#D97706]/30",
    accent: "text-[#D97706]",
  },
];

export function QuickActions() {
  return (
    <section className="cl-card p-5 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3.5">
        <div className="w-6 h-6 rounded-lg bg-[#E0F9E6] text-[#158444] flex items-center justify-center shrink-0">
          <Zap className="w-3.5 h-3.5 fill-[#158444]" />
        </div>
        <h3 className="text-[14.5px] font-bold text-[#111613] tracking-tight">Quick Actions</h3>
      </div>

      {/* 2x2 Grid of Actions */}
      <div className="grid grid-cols-2 gap-2.5">
        {ACTIONS.map(({ title, href, icon: Icon, iconBg, hoverBg, hoverBorder, accent }) => (
          <Link
            key={title}
            href={href}
            className={`group bg-white ${hoverBg} border border-[#E8EDE9] ${hoverBorder} rounded-xl p-3 sm:p-3.5 flex flex-col items-center justify-between text-center transition-all shadow-2xs hover:shadow-xs min-h-[102px]`}
          >
            <span className={`w-8 h-8 rounded-lg ${iconBg} flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform`}>
              <Icon className="w-4 h-4 stroke-[2.2]" />
            </span>
            <span className="text-[11.5px] sm:text-[12px] font-bold text-[#111613] leading-snug mt-1.5 transition-colors">
              {title}
            </span>
            <ArrowRight className={`w-3.5 h-3.5 ${accent} mt-1 group-hover:translate-x-0.5 transition-transform stroke-[2.5]`} />
          </Link>
        ))}
      </div>
    </section>
  );
}
