"use client";

import React from "react";
import { Code2 } from "lucide-react";

interface TechItem {
  category: string;
  technologies: string;
  iconType: "spark" | "python" | "shap" | "node" | "react" | "mongo" | "docker";
}

const TECH_ITEMS: TechItem[] = [
  {
    category: "Data Processing",
    technologies: "Apache Spark",
    iconType: "spark",
  },
  {
    category: "Machine Learning",
    technologies: "Python • scikit-learn • XGBoost",
    iconType: "python",
  },
  {
    category: "Explainability",
    technologies: "SHAP",
    iconType: "shap",
  },
  {
    category: "Backend",
    technologies: "Node.js • Express",
    iconType: "node",
  },
  {
    category: "Frontend",
    technologies: "React (Vite) • Tailwind CSS",
    iconType: "react",
  },
  {
    category: "Database / Storage",
    technologies: "MongoDB",
    iconType: "mongo",
  },
  {
    category: "Deployment",
    technologies: "Docker • Vercel",
    iconType: "docker",
  },
];

function TechLogo({ type }: { type: TechItem["iconType"] }) {
  switch (type) {
    case "spark":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#E25A1C] font-bold text-[11px]">
          ⚡
        </span>
      );
    case "python":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#3776AB] font-bold text-[11px]">
          🐍
        </span>
      );
    case "shap":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#D83B01] font-bold text-[11px]">
          📊
        </span>
      );
    case "node":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#5FA04E] font-bold text-[11px]">
          🟢
        </span>
      );
    case "react":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#00D8FF] font-bold text-[11px]">
          ⚛️
        </span>
      );
    case "mongo":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#47A248] font-bold text-[11px]">
          🍃
        </span>
      );
    case "docker":
      return (
        <span className="w-4 h-4 inline-flex items-center justify-center shrink-0 text-[#2496ED] font-bold text-[11px]">
          🐳
        </span>
      );
    default:
      return null;
  }
}

export function TechStackCard() {
  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Code2 className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Tech Stack
        </h2>
      </div>

      {/* Rows */}
      <div className="space-y-2.5">
        {TECH_ITEMS.map((item) => (
          <div
            key={item.category}
            className="flex items-center justify-between gap-2 py-0.5"
          >
            {/* Category Pill */}
            <span className="px-2.5 py-0.5 rounded-full bg-[#EAF5EE] text-[#158444] text-[11px] font-semibold whitespace-nowrap shrink-0">
              {item.category}
            </span>

            {/* Tech Logo & Name */}
            <div className="flex items-center gap-1.5 text-right min-w-0">
              <TechLogo type={item.iconType} />
              <span className="text-[12px] font-medium text-[#111613] truncate">
                {item.technologies}
              </span>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}
