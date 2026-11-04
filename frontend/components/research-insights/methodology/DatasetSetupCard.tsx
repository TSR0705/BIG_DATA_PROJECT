"use client";

import React from "react";
import { Database } from "lucide-react";

interface SetupItem {
  label: string;
  value: string;
}

const ITEMS: SetupItem[] = [
  { label: "Dataset", value: "Home Credit Default Risk" },
  { label: "Total Records", value: "307,511" },
  { label: "Total Features", value: "240" },
  { label: "Train : Validation : Test", value: "70% : 15% : 15%" },
  { label: "Processing Framework", value: "Apache Spark" },
  { label: "Environment", value: "Python, PySpark, scikit-learn" },
];

export function DatasetSetupCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2.5 mb-4">
          <Database className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Dataset &amp; Setup</h2>
        </div>

        {/* 6 Key-Value Rows */}
        <div className="divide-y divide-[#F2F4F2] mt-2">
          {ITEMS.map((item) => (
            <div key={item.label} className="flex items-center justify-between py-2.5">
              <span className="text-[12px] font-medium text-[#555E57]">
                {item.label}
              </span>
              <span className="text-[12px] font-bold text-[#111613] tabular-nums text-right">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}
