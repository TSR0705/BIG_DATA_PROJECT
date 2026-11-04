"use client";

import React from "react";
import { Check, Lightbulb } from "lucide-react";

interface TakeawayItem {
  title: string;
  description: string;
}

const DEFAULT_TAKEAWAYS: TakeawayItem[] = [
  {
    title: "Main Reason",
    description: "The prediction is mainly influenced by loan amount and age.",
  },
  {
    title: "Risk Level",
    description: "This is a moderate risk (14.2%). Additional checks may be considered.",
  },
  {
    title: "Reliability",
    description: "The model is well-calibrated and provides a reliable probability.",
  },
  {
    title: "Next Step",
    description: "Use this insight as a decision support, not a final approval/rejection.",
  },
];

interface Props {
  items?: TakeawayItem[];
}

export function KeyTakeawaysCard({ items = DEFAULT_TAKEAWAYS }: Props) {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2.5 mb-5">
          <Lightbulb className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Key Takeaways</h2>
        </div>

        {/* 4 Takeaway rows */}
        <div className="space-y-4">
          {items.map((item) => (
            <div key={item.title} className="flex items-start gap-3">
              <div className="w-5 h-5 rounded-full bg-[#158444] text-white flex items-center justify-center shrink-0 mt-0.5">
                <Check className="w-3 h-3 stroke-[3]" />
              </div>
              <div className="min-w-0">
                <h3 className="text-[13px] font-bold text-[#111613] leading-snug">
                  {item.title}
                </h3>
                <p className="text-[12px] text-[#555E57] mt-0.5 leading-relaxed">
                  {item.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}
