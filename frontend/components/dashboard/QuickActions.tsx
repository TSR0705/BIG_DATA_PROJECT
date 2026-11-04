import Link from "next/link";
import { ArrowRight, BarChart3, BookOpen, Database, UserPlus, Zap, type LucideIcon } from "lucide-react";

const ACTIONS: { title: string; href: string; icon: LucideIcon }[] = [
  { title: "Check Loan Risk", href: "/check-risk", icon: UserPlus },
  { title: "View Model Performance", href: "/model-performance", icon: BarChart3 },
  { title: "Explore Dataset", href: "/data-pipeline", icon: Database },
  { title: "See Research", href: "/research-insights", icon: BookOpen },
];

export function QuickActions() {
  return (
    <section className="cl-card p-5 flex flex-col">
      <div className="flex items-center gap-2 mb-3.5">
        <Zap className="w-4 h-4 text-[#158444] fill-[#158444]" />
        <h3 className="text-[15px] font-bold text-ink">Quick Actions</h3>
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        {ACTIONS.map(({ title, href, icon: Icon }) => (
          <Link
            key={title}
            href={href}
            className="group bg-white hover:bg-[#EBF6F0]/40 border border-[#E9ECE9] hover:border-[#158444]/30 rounded-xl p-3 flex flex-col items-center justify-between text-center transition-all shadow-2xs min-h-[96px]"
          >
            <Icon className="w-5 h-5 text-[#158444] stroke-[2] group-hover:scale-105 transition-transform" />
            <span className="text-[11.5px] font-semibold text-ink leading-tight mt-1 group-hover:text-[#158444] transition-colors">{title}</span>
            <ArrowRight className="w-3.5 h-3.5 text-[#158444] mt-1 group-hover:translate-x-1 transition-transform stroke-[2.5]" />
          </Link>
        ))}
      </div>
    </section>
  );
}
