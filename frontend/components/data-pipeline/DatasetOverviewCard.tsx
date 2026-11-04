"use client";

import React from "react";
import { Database } from "lucide-react";

interface OverviewItem {
  label: string;
  value: string;
}

const ITEMS: OverviewItem[] = [
  { label: "Dataset", value: "Home Credit Default Risk" },
  { label: "Total Records", value: "307,511" },
  { label: "Total Features", value: "240" },
  { label: "Default Rate", value: "8.07%" },
  { label: "Data Size", value: "~ 1.8 GB" },
  { label: "Format", value: "CSV / Parquet (Spark)" },
];

export function DatasetOverviewCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <Database className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
          <div>
            <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
              Dataset Overview
            </h2>
            <p className="text-[11.5px] text-[#89938D] mt-0.5">
              Key information about the training dataset.
            </p>
          </div>
        </div>

        {/* 6 Key-Value Rows */}
        <div className="mt-4 divide-y divide-[#F2F4F2]">
          {ITEMS.map((item) => (
            <div key={item.label} className="flex items-center justify-between py-2.5">
              <span className="text-[12.5px] font-medium text-[#555E57]">
                {item.label}
              </span>
              <span className="text-[12.5px] font-bold text-[#111613] tabular-nums">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}
