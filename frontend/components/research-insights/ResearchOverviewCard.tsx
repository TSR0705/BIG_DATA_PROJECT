"use client";

import React from "react";
import { BookOpen } from "lucide-react";

const TAGS = ["Big Data", "Machine Learning", "XGBoost", "SHAP", "Credit Risk"];

export function ResearchOverviewCard() {
  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2.5 mb-3">
          <BookOpen className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
          <h2 className="text-[17px] font-bold text-[#111613]">Research Overview</h2>
        </div>

        {/* Description */}
        <p className="text-[13px] text-[#555E57] leading-relaxed mt-2">
          A Big Data–driven approach to predict loan default using the Home Credit Default Risk dataset.
          The study combines scalable data processing with machine learning and explainable AI to
          provide accurate and interpretable credit risk predictions.
        </p>
      </div>

      {/* 5 Tag Pills */}
      <div className="flex flex-wrap items-center gap-2 mt-6 pt-2">
        {TAGS.map((tag) => (
          <span
            key={tag}
            className="inline-flex items-center px-3.5 py-1 rounded-full bg-[#EAF5EE] text-[#158444] text-[11.5px] font-semibold"
          >
            {tag}
          </span>
        ))}
      </div>
    </article>
  );
}
