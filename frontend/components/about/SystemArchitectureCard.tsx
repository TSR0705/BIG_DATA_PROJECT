"use client";

import React from "react";
import { BarChart3, Database, Layers, Monitor, Settings, Share2 } from "lucide-react";

interface ArchNode {
  title: string;
  subtitle: string;
  icon: React.ElementType;
  isHighlighted?: boolean;
}

const ARCH_NODES: ArchNode[] = [
  {
    title: "Data Source",
    subtitle: "(Home Credit Dataset)",
    icon: Database,
  },
  {
    title: "Data Pipeline",
    subtitle: "(Apache Spark Processing)",
    icon: Settings,
  },
  {
    title: "ML Models",
    subtitle: "(XGBoost, LightGBM, etc.)",
    icon: Layers,
  },
  {
    title: "Explainability",
    subtitle: "(SHAP)",
    icon: BarChart3,
  },
  {
    title: "Web Application",
    subtitle: "(Prediction & Insights)",
    icon: Monitor,
    isHighlighted: true,
  },
];

export function SystemArchitectureCard() {
  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3.5">
        <Share2 className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          System Architecture
        </h2>
      </div>

      {/* 5-Node Flowchart */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-2 sm:gap-1.5 overflow-x-auto py-1">
        {ARCH_NODES.map((node, index) => {
          const Icon = node.icon;
          return (
            <React.Fragment key={node.title}>
              {/* Stage Box */}
              <div
                className={`flex-1 w-full sm:w-auto p-3 rounded-xl border text-center flex flex-col items-center justify-center min-w-[105px] transition-all ${
                  node.isHighlighted
                    ? "bg-[#EAF5EE] border-[#A7E2BA]"
                    : "bg-white border-[#E2E8E2]"
                }`}
              >
                <Icon className="w-5 h-5 text-[#158444] mb-1.5 stroke-[2.2]" />
                <p className="text-[11.5px] sm:text-[12px] font-bold text-[#111613] tracking-tight leading-tight">
                  {node.title}
                </p>
                <p className="text-[9.5px] sm:text-[10px] text-[#555E57] mt-0.5 leading-tight">
                  {node.subtitle}
                </p>
              </div>

              {/* Connector Arrow */}
              {index < ARCH_NODES.length - 1 && (
                <div className="text-[#158444] font-bold text-base sm:text-lg select-none px-0.5">
                  →
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </article>
  );
}
