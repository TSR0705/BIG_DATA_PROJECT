"use client";

import React from "react";
import { Database } from "lucide-react";

export function DatasetCard() {
  const metadata = [
    { label: "Name", value: "Home Credit Default Risk" },
    { label: "Records", value: "307,511" },
    { label: "Features", value: "240" },
    { label: "Target", value: "Loan Default (0 = No, 1 = Yes)" },
    { label: "Source", value: "Kaggle (Home Credit)" },
  ];

  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Database className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Dataset
        </h2>
      </div>

      {/* Key-Value Table */}
      <div className="space-y-3 divide-y divide-[#F5F7F5]">
        {metadata.map((item, index) => (
          <div
            key={item.label}
            className={`flex items-center justify-between gap-3 ${
              index > 0 ? "pt-2.5" : ""
            }`}
          >
            <span className="text-[12px] font-bold text-[#111613] shrink-0">
              {item.label}
            </span>
            <span className="text-[12px] text-[#555E57] text-right truncate">
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </article>
  );
}
