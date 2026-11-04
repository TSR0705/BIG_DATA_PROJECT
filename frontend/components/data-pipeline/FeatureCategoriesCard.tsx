"use client";

import React from "react";
import { Layers } from "lucide-react";

interface CategoryItem {
  name: string;
  pct: number;
  color: string;
}

const CATEGORIES: CategoryItem[] = [
  { name: "Credit & Payment History", pct: 32, color: "#10543A" },
  { name: "Loan & Contract Details", pct: 24, color: "#20865D" },
  { name: "Applicant Demographics", pct: 18, color: "#48C774" },
  { name: "Financial Behaviour", pct: 16, color: "#86EFAC" },
  { name: "External & Derived Features", pct: 10, color: "#BBF7D0" },
];

export function FeatureCategoriesCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <Layers className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Feature Categories
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Engineered feature groups used by the model.
            </p>
          </div>
        </div>

        {/* 5 Categories */}
        <div className="mt-5 space-y-4">
          {CATEGORIES.map((cat) => (
            <div
              key={cat.name}
              className="grid grid-cols-[minmax(0,10.5rem)_1fr_auto] items-center gap-3"
            >
              <span className="text-[12.5px] font-medium text-[#111613] truncate">
                {cat.name}
              </span>

              <div className="h-3 rounded-full bg-[#EEF2EE] overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${cat.pct}%`,
                    backgroundColor: cat.color,
                  }}
                />
              </div>

              <span className="text-[12.5px] font-bold text-[#111613] tabular-nums w-10 text-right">
                {cat.pct}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}
