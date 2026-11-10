"use client";

import React from "react";
import { Database, Layers, Lightbulb, Search, ShieldCheck } from "lucide-react";

interface ChoiceItem {
  title: string;
  description: string;
  icon: React.ElementType;
}

const CHOICES: ChoiceItem[] = [
  {
    title: "Big Data Processing",
    description: "Used Apache Spark for scalable and efficient data handling.",
    icon: Database,
  },
  {
    title: "Feature Engineering",
    description: "Created domain-specific credit risk features.",
    icon: Layers,
  },
  {
    title: "Model Selection",
    description: "Compared multiple ML models and chosen XGBoost.",
    icon: ShieldCheck,
  },
  {
    title: "Model Explainability",
    description: "Used SHAP to provide transparent and interpretable results.",
    icon: Search,
  },
];

export function KeyDesignChoicesCard() {
  return (
    <article className="cl-card p-5 lg:p-6 bg-white">
      {/* Header */}
      <div className="flex items-center gap-2.5 mb-5">
        <Lightbulb className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0" />
        <h2 className="text-[17px] font-bold text-[#111613]">Key Design Choices</h2>
      </div>

      {/* 4 Items Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {CHOICES.map((c) => {
          const Icon = c.icon;
          return (
            <div key={c.title} className="flex items-start gap-3.5">
              <Icon className="w-6 h-6 text-[#158444] stroke-[2] shrink-0 mt-0.5" />
              <div>
                <h3 className="text-[13.5px] font-bold text-[#111613] leading-snug">
                  {c.title}
                </h3>
                <p className="text-[12px] text-[#555E57] mt-1 leading-snug">
                  {c.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}
