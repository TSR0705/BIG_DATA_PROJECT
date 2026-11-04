"use client";

import React from "react";
import { PipelineKpiRow } from "@/components/data-pipeline/PipelineKpiRow";
import { PipelineFlowCard } from "@/components/data-pipeline/PipelineFlowCard";
import { DatasetOverviewCard } from "@/components/data-pipeline/DatasetOverviewCard";
import { DataQualityCard } from "@/components/data-pipeline/DataQualityCard";
import { FeatureCategoriesCard } from "@/components/data-pipeline/FeatureCategoriesCard";

export default function DataPipelinePage() {
  return (
    <div className="space-y-5">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl sm:text-[26px] font-extrabold text-[#111613] tracking-tight">
          Data & Pipeline
        </h1>
        <p className="text-[13px] text-[#555E57] mt-1">
          Explore the dataset, data processing pipeline, and feature engineering used for loan default prediction.
        </p>
      </div>

      {/* Row 1: 4 Top KPI Cards */}
      <PipelineKpiRow />

      {/* Row 2: Data Processing Pipeline Flowchart */}
      <PipelineFlowCard />

      {/* Row 3: 3 Cards Across (Dataset Overview, Data Quality, Feature Categories) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
        <DatasetOverviewCard />
        <DataQualityCard />
        <FeatureCategoriesCard />
      </div>
    </div>
  );
}
