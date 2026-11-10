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
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none">
          <path
            d="M12 2L14.5 8.5L21 9L16 13.5L17.5 20L12 16.5L6.5 20L8 13.5L3 9L9.5 8.5L12 2Z"
            fill="#E25A1C"
          />
        </svg>
      );
    case "python":
      return (
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none">
          <path
            d="M11.9 2C6.8 2 7.1 4.2 7.1 4.2L7.2 6.5H12.1V7.2H5.1C2.9 7.2 2 9.5 2 11.9C2 14.8 3.5 15.6 5.1 15.6H6.7V13.7C6.7 11.3 8.7 11.3 8.7 11.3H13.6C15.2 11.3 16.4 10 16.4 8.5V4.2C16.4 2.8 14.9 2 11.9 2ZM9.6 3.6C10.2 3.6 10.7 4.1 10.7 4.7C10.7 5.3 10.2 5.8 9.6 5.8C9 5.8 8.5 5.3 8.5 4.7C8.5 4.1 9 3.6 9.6 3.6Z"
            fill="#3776AB"
          />
          <path
            d="M12.1 22C17.2 22 16.9 19.8 16.9 19.8L16.8 17.5H11.9V16.8H18.9C21.1 16.8 22 14.5 22 12.1C22 9.2 20.5 8.4 18.9 8.4H17.3V10.3C17.3 12.7 15.3 12.7 15.3 12.7H10.4C8.8 12.7 7.6 14 7.6 15.5V19.8C7.6 21.2 9.1 22 12.1 22ZM14.4 20.4C13.8 20.4 13.3 19.9 13.3 19.3C13.3 18.7 13.8 18.2 14.4 18.2C15 18.2 15.5 18.7 15.5 19.3C15.5 19.9 15 20.4 14.4 20.4Z"
            fill="#FFD43B"
          />
        </svg>
      );
    case "shap":
      return (
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="14" width="3" height="7" rx="1" fill="#FF4136" />
          <rect x="8" y="10" width="3" height="11" rx="1" fill="#FF851B" />
          <rect x="13" y="6" width="3" height="15" rx="1" fill="#2ECC40" />
          <rect x="18" y="3" width="3" height="18" rx="1" fill="#0074D9" />
        </svg>
      );
    case "node":
      return (
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="#5FA04E">
          <path d="M12 2L21 7.2V16.8L12 22L3 16.8V7.2L12 2ZM12 4.3L5 8.3V15.7L12 19.7L19 15.7V8.3L12 4.3Z" />
        </svg>
      );
    case "react":
      return (
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="#00D8FF" strokeWidth="1.5">
          <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(0 12 12)" />
          <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(60 12 12)" />
          <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(120 12 12)" />
          <circle cx="12" cy="12" r="1.5" fill="#00D8FF" />
        </svg>
      );
    case "mongo":
      return (
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="#47A248">
          <path d="M12 2C12 2 6 7.5 6 13C6 16.5 8.5 19.5 12 22C15.5 19.5 18 16.5 18 13C18 7.5 12 2 12 2ZM12 19.5C9.5 17.5 8 15 8 13C8 9.5 12 5.5 12 5.5C12 5.5 16 9.5 16 13C16 15 14.5 17.5 12 19.5Z" />
        </svg>
      );
    case "docker":
      return (
        <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="#2496ED">
          <path d="M2 13C3.5 13 4.5 14 6 14C7.5 14 8.5 13 10 13C11.5 13 12.5 14 14 14C15.5 14 16.5 13 18 13C19.5 13 20.5 14 22 14C22 18 19 21 12 21C5 21 2 18 2 13ZM5 9H8V12H5V9ZM9 9H12V12H9V9ZM13 9H16V12H13V9ZM9 5H12V8H9V5ZM13 5H16V8H13V5ZM17 9H20V12H17V9Z" />
        </svg>
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
