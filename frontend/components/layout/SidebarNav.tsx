"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_ITEMS, SECONDARY_NAV, type NavItem } from "./nav";
import { BrandMark } from "./BrandMark";

function NavLink({ item, onNavigate }: { item: NavItem; onNavigate?: () => void }) {
  const pathname = usePathname();
  const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      onClick={onNavigate}
      aria-current={isActive ? "page" : undefined}
      className={cn(
        "flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-[13px] font-medium transition-all",
        isActive
          ? "bg-[#E0F9E6] text-[#087A43] font-bold shadow-2xs"
          : "text-[#555E57] hover:bg-[#F2F4F2] hover:text-[#111613]",
      )}
    >
      <Icon className={cn("w-4 h-4 stroke-[2.2]", isActive ? "text-[#087A43]" : "text-[#89938D]")} />
      <span>{item.name}</span>
    </Link>
  );
}

/** Navigation content shared by the desktop sidebar and the mobile drawer. */
export function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex flex-col justify-between h-full p-4">
      <div>
        <div className="mb-6">
          <BrandMark />
        </div>
        <nav className="space-y-1" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.href} item={item} onNavigate={onNavigate} />
          ))}
        </nav>
        <div className="my-5 border-t border-line" />
        <nav className="space-y-1" aria-label="Secondary">
          {SECONDARY_NAV.map((item) => (
            <NavLink key={item.href} item={item} onNavigate={onNavigate} />
          ))}
        </nav>
      </div>

      {/* Promo card */}
      <div className="bg-gradient-to-br from-[#EAF8EF] to-[#DDF3E5] border border-[#CDEBD7] rounded-2xl p-4 mt-6 shadow-2xs">
        <div className="w-11 h-11 rounded-xl bg-white border border-[#DEEDE2] flex items-center justify-center mb-3 shadow-2xs">
          <svg className="w-7 h-7" viewBox="0 0 32 32" aria-hidden="true">
            <rect x="6" y="18" width="4" height="8" rx="1.5" fill="#48C774" />
            <rect x="14" y="12" width="4" height="14" rx="1.5" fill="#20865D" />
            <rect x="22" y="8" width="4" height="18" rx="1.5" fill="#10543A" />
            <path d="M 6 15 L 14 10 L 22 5" fill="none" stroke="#10543A" strokeWidth="2" strokeLinecap="round" />
            <polygon points="20,4 24,5 23,9" fill="#10543A" />
          </svg>
        </div>
        <h4 className="font-bold text-[13px] text-[#111613] leading-snug">Big Data Analytics for Credit Risk</h4>
        <p className="text-[11px] text-[#555E57] mt-1.5 leading-relaxed font-medium">
          Using Spark and machine learning to analyse 300K+ loan applications.
        </p>
        <Link
          href="/about"
          onClick={onNavigate}
          className="mt-3.5 inline-flex items-center justify-center gap-1.5 w-full bg-[#087A43] hover:bg-[#066336] text-white text-xs font-semibold py-2.5 px-3 rounded-full transition-all shadow-xs"
        >
          <span>Learn More</span>
          <ArrowRight className="w-3.5 h-3.5 stroke-[2.5]" />
        </Link>
      </div>
    </div>
  );
}
