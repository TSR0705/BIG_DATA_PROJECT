"use client";

import React from "react";
import {
  BarChart3,
  Clock,
  CloudUpload,
  Cpu,
  Database,
  Layers,
  Scale,
} from "lucide-react";

interface FutureDirection {
  title: string;
  icon: React.ElementType;
  description: string;
}

const DIRECTIONS: FutureDirection[] = [
  {
    title: "Richer & Multimodal Data",
    icon: Database,
    description:
      "Incorporate additional data sources (e.g., transaction history, alternative credit data, macroeconomic indicators) to improve prediction.",
  },
  {
    title: "Advanced Modeling",
    icon: Cpu,
    description:
      "Explore ensemble and deep learning approaches (e.g., stacking, TabNet, or temporal models) for higher accuracy.",
  },
  {
    title: "Fairness & Explainability",
    icon: Scale,
    description:
      "Evaluate and mitigate bias (e.g., demographic parity, equal opportunity) and enhance explanations with more user-friendly insights.",
  },
  {
    title: "Real-Time Prediction",
    icon: Clock,
    description:
      "Design a real-time inference pipeline for on-demand loan risk scoring and monitoring.",
  },
  {
    title: "Scalable Deployment",
    icon: CloudUpload,
    description:
      "Deploy as a production-ready, secure, and scalable service (e.g., Docker + Kubernetes on multi-cloud).",
  },
  {
    title: "Broader Evaluation",
    icon: BarChart3,
    description:
      "Validate on additional datasets, perform long-term A/B testing, and assess business impact (e.g., reduced default rate, ROI).",
  },
];

export function FutureWorkDirectionsGrid() {
  return (
    <section>
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-3.5 sm:mb-4">
        <Layers className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Key Future Work Directions
        </h2>
      </div>

      {/* 3x2 Grid of Direction Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 items-stretch">
        {DIRECTIONS.map((dir) => {
          const Icon = dir.icon;
          return (
            <article
              key={dir.title}
              className="cl-card p-5 sm:p-5.5 bg-white border border-[#E8EEE9] flex items-start gap-3.5"
            >
              {/* Rounded Green Outlined Icon Badge */}
              <div className="w-10 h-10 rounded-full border border-[#D1E7D8] bg-[#F2FAF4] flex items-center justify-center shrink-0 mt-0.5">
                <Icon className="w-5 h-5 text-[#158444] stroke-[2.2]" />
              </div>

              {/* Text Info */}
              <div className="min-w-0">
                <h3 className="text-[13.5px] sm:text-[14px] font-bold text-[#111613] tracking-tight">
                  {dir.title}
                </h3>
                <p className="text-[12px] text-[#555E57] mt-1 leading-relaxed">
                  {dir.description}
                </p>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
