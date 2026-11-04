"use client";

import React from "react";
import {
  ArrowRight,
  BarChart3,
  Box,
  Code2,
  Cog,
  Database,
  Layers,
  Share2,
} from "lucide-react";

interface PipelineStep {
  name: string;
  description: string;
  icon: React.ElementType;
  isFinal?: boolean;
}

const STEPS: PipelineStep[] = [
  {
    name: "Data Source",
    description: "Home Credit Dataset (300K+ records)",
    icon: Database,
  },
  {
    name: "Ingestion",
    description: "Load data using Apache Spark",
    icon: Cog,
  },
  {
    name: "Preprocessing",
    description: "Data cleaning, missing value handling",
    icon: Layers,
  },
  {
    name: "Feature Engineering",
    description: "Create risk and behavioral features",
    icon: Code2,
  },
  {
    name: "Model Training",
    description: "Train & calibrate (XGBoost)",
    icon: Box,
  },
  {
    name: "Prediction",
    description: "Estimate default risk & explain results",
    icon: BarChart3,
    isFinal: true,
  },
];

export function PipelineFlowCard() {
  return (
    <article className="cl-card p-5 lg:p-6 bg-white">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <div className="flex items-start gap-3">
          <Share2 className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Data Processing Pipeline
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              End-to-end flow from raw data to model prediction.
            </p>
          </div>
        </div>

        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#EAF5EE] text-[#158444] text-[12px] font-semibold self-start sm:self-auto">
          <span className="w-2 h-2 rounded-full bg-[#158444] animate-pulse" />
          <span>Pipeline Ready</span>
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

                {/* Desktop Arrow Connector pointing to next stage */}
                {idx < STEPS.length - 1 && (
                  <div className="hidden xl:flex absolute -right-3 top-1/2 -translate-y-1/2 z-10 w-6 h-6 items-center justify-center pointer-events-none">
                    <ArrowRight className="w-4 h-4 text-[#89938D] stroke-[2.2]" />
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
