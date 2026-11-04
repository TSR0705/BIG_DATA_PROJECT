"use client";

import React from "react";
import { Calendar, Check, Target } from "lucide-react";

interface RoadmapStep {
  step: number;
  phase: string;
  duration: string;
  description: string;
  circleBg: string;
}

const ROADMAP_STEPS: RoadmapStep[] = [
  {
    step: 1,
    phase: "Short Term",
    duration: "(1–2 months)",
    description: "Add new features, try advanced models.",
    circleBg: "bg-[#48C774]",
  },
  {
    step: 2,
    phase: "Mid Term",
    duration: "(3–6 months)",
    description: "Fairness evaluation, model improvement, real-time inference.",
    circleBg: "bg-[#20865D]",
  },
  {
    step: 3,
    phase: "Long Term",
    duration: "(6–12 months)",
    description: "Deploy a scalable system on cloud multi-platform.",
    circleBg: "bg-[#158444]",
  },
  {
    step: 4,
    phase: "Future Research",
    duration: "(> 12 months)",
    description: "A/B testing, broader datasets, and business impact study.",
    circleBg: "bg-[#0E5E3D]",
  },
];

const IMPACT_POINTS = [
  {
    lead: "Higher accuracy",
    rest: " and better generalization on diverse data.",
  },
  {
    lead: "Fairer",
    rest: " and more transparent lending decisions.",
  },
  {
    lead: "Real-time, scalable,",
    rest: " and production-ready system.",
  },
  {
    lead: "Measurable business value",
    rest: " (e.g., lower default rate, risk reduction).",
  },
];

export function RoadmapAndImpactRow() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 items-stretch">
      {/* Left Card: Proposed Roadmap */}
      <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between lg:col-span-7">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
            <h3 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
              Proposed Roadmap
            </h3>
          </div>
          <p className="text-[12px] text-[#555E57] mt-0.5">
            A phased plan to extend and operationalize the research.
          </p>
        </div>

        {/* Stepper Timeline */}
        <div className="mt-6 relative">
          {/* Horizontal Connecting Line */}
          <div className="hidden sm:block absolute top-3.5 left-8 right-8 h-0.5 bg-[#158444] z-0" />

          {/* Steps */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 relative z-10">
            {ROADMAP_STEPS.map((step) => (
              <div key={step.step} className="flex flex-col items-center text-center">
                {/* Step Circle Badge */}
                <div
                  className={`w-7 h-7 rounded-full ${step.circleBg} text-white flex items-center justify-center font-bold text-[12px] shadow-xs shrink-0`}
                >
                  {step.step}
                </div>

                {/* Phase Title */}
                <p className="text-[12px] font-bold text-[#111613] mt-2 leading-tight">
                  {step.phase}
                </p>

                {/* Duration */}
                <p className="text-[11px] font-medium text-[#555E57] mt-0.5">
                  {step.duration}
                </p>

                {/* Description */}
                <p className="text-[10.5px] sm:text-[11px] text-[#555E57] mt-1.5 leading-snug">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </article>

      {/* Right Card: Expected Impact */}
      <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between lg:col-span-5">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
            <h3 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
              Expected Impact
            </h3>
          </div>
          <p className="text-[12px] text-[#555E57] mt-0.5">
            Potential benefits of the planned future work.
          </p>
        </div>

        {/* Impact Bullet Points */}
        <div className="mt-4 space-y-3 sm:space-y-3.5">
          {IMPACT_POINTS.map((pt, idx) => (
            <div key={idx} className="flex items-start gap-2.5">
              <span className="w-4 h-4 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                <Check className="w-2.5 h-2.5 stroke-[3]" />
              </span>
              <p className="text-[12px] sm:text-[12.5px] text-[#222924] leading-relaxed">
                <strong className="font-bold text-[#111613]">{pt.lead}</strong>
                {pt.rest}
              </p>
            </div>
          ))}
        </div>
      </article>
    </div>
  );
}
