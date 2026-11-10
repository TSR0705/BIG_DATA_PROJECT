"use client";

import React from "react";
import {
  ArrowRight,
  BarChart3,
  Cog,
  Database,
  Layers,
  Share2,
  Target,
} from "lucide-react";

interface MethodologyStep {
  name: string;
  description: string;
  icon: React.ElementType;
  isFinal?: boolean;
}

const STEPS: MethodologyStep[] = [
  {
    name: "Data Collection",
    description: "Home Credit Dataset (300K+ records)",
    icon: Database,
  },
  {
    name: "Data Preprocessing",
    description: "Cleaning, missing value handling & encoding",
    icon: Cog,
  },
  {
    name: "Feature Engineering",
    description: "Create financial & behavioral features",
    icon: Layers,
  },
  {
    name: "Modeling",
    description: "Train ML models (XGBoost, LightGBM, etc.)",
    icon: Share2,
  },
  {
    name: "Evaluation",
    description: "Assess using ROC-AUC, PR-AUC, Brier, ECE",
    icon: Target,
  },
  {
    name: "Interpretation",
    description: "Explain results using SHAP",
    icon: BarChart3,
    isFinal: true,
  },
];

export function MethodologyOverviewFlowCard() {
  return (
    <article className="cl-card p-5 lg:p-6 bg-white">
      {/* Header */}
      <div className="flex items-start gap-3 mb-6">
        <Layers className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
        <div>
          <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
            Methodology Overview
          </h2>
          <p className="text-[11.5px] text-[#89938D] mt-0.5">
            A structured end-to-end approach combining scalable data processing and machine learning to predict loan default risk.
          </p>
        </div>
      </div>

      {/* 6 Stages Grid with Arrow Connectors */}
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3 sm:gap-2 items-center">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isFinal = step.isFinal;
          return (
            <React.Fragment key={step.name}>
              <div
                className={`relative flex flex-col items-center justify-center text-center p-4 rounded-2xl border transition-all h-[155px] ${
                  isFinal
                    ? "bg-[#EAF5EE] border-[#C6F2D2] shadow-2xs"
                    : "bg-white border-[#E9ECE9] hover:border-[#D5DDD7]"
                }`}
              >
                {/* Icon */}
                <div className="w-10 h-10 rounded-xl flex items-center justify-center text-[#158444] mb-2.5">
                  <Icon className="w-6 h-6 stroke-[2]" />
                </div>

                {/* Title */}
                <h3 className="text-[13px] font-bold text-[#111613] leading-tight mb-1">
                  {step.name}
                </h3>

                {/* Description */}
                <p className="text-[11px] text-[#555E57] leading-tight line-clamp-2 px-1">
                  {step.description}
                </p>

                {/* Desktop Arrow Connector */}
                {idx < STEPS.length - 1 && (
                  <div className="hidden xl:flex absolute -right-3 top-1/2 -translate-y-1/2 z-10 w-6 h-6 items-center justify-center pointer-events-none">
                    <ArrowRight className="w-4 h-4 text-[#158444] stroke-[2.4]" />
                  </div>
                )}
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </article>
  );
}
