"use client";

import React from "react";
import { AlertTriangle, BarChart3, FileText, Layers, Rocket } from "lucide-react";

export type ResearchTabId = "overview" | "methodology" | "findings" | "limitations" | "future_work";

interface TabItem {
  id: ResearchTabId;
  label: string;
  icon: React.ElementType;
}

const TABS: TabItem[] = [
  { id: "overview", label: "Overview", icon: FileText },
  { id: "methodology", label: "Methodology", icon: Layers },
  { id: "findings", label: "Findings", icon: BarChart3 },
  { id: "limitations", label: "Limitations", icon: AlertTriangle },
  { id: "future_work", label: "Future Work", icon: Rocket },
];

interface Props {
  activeTab: ResearchTabId;
  onTabChange: (id: ResearchTabId) => void;
}

export function ResearchTabsRow({ activeTab, onTabChange }: Props) {
  return (
    <div className="flex items-center gap-2.5 sm:gap-3 overflow-x-auto pb-1 scrollbar-none select-none">
      {TABS.map((t) => {
        const Icon = t.icon;
        const isActive = activeTab === t.id;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onTabChange(t.id)}
            className={`inline-flex items-center gap-2 px-5 py-2 rounded-full text-[13px] font-semibold transition-all shrink-0 cursor-pointer ${
              isActive
                ? "bg-[#158444] text-white shadow-xs"
                : "bg-white text-[#555E57] border border-[#D5DDD7] hover:bg-[#FAFBF9] hover:text-[#111613]"
            }`}
          >
            <Icon className={`w-4 h-4 ${isActive ? "text-white stroke-[2.2]" : "text-[#555E57] stroke-[2]"}`} />
            <span>{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}
