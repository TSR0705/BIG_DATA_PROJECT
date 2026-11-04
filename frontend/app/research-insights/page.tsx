"use client";

import React, { useState } from "react";
import { ResearchTabsRow, type ResearchTabId } from "@/components/research-insights/ResearchTabsRow";
import { KeyFindingsRow } from "@/components/research-insights/KeyFindingsRow";
import { ResearchOverviewCard } from "@/components/research-insights/ResearchOverviewCard";
import { OurContributionsCard } from "@/components/research-insights/OurContributionsCard";
import { ModelComparisonResearchCard } from "@/components/research-insights/ModelComparisonResearchCard";
import { ReadResearchCard } from "@/components/research-insights/ReadResearchCard";

// Methodology Tab Components
import { MethodologyOverviewFlowCard } from "@/components/research-insights/methodology/MethodologyOverviewFlowCard";
import { ModelingApproachCard } from "@/components/research-insights/methodology/ModelingApproachCard";
import { DatasetSetupCard } from "@/components/research-insights/methodology/DatasetSetupCard";
import { EvaluationMetricsCard } from "@/components/research-insights/methodology/EvaluationMetricsCard";
import { KeyDesignChoicesCard } from "@/components/research-insights/methodology/KeyDesignChoicesCard";

// Findings Tab Components
import { FindingsKpiRow } from "@/components/research-insights/findings/FindingsKpiRow";
import { ModelComparisonTestCard } from "@/components/research-insights/findings/ModelComparisonTestCard";
import { MultiRocCurveCard } from "@/components/research-insights/findings/MultiRocCurveCard";
import { FeatureImportanceShapCard } from "@/components/research-insights/findings/FeatureImportanceShapCard";
import { FindingsKeyInsightsCard } from "@/components/research-insights/findings/FindingsKeyInsightsCard";

// Limitations Tab Components
import { LimitationsHeaderRow } from "@/components/research-insights/limitations/LimitationsHeaderRow";
import { LimitationsGrid } from "@/components/research-insights/limitations/LimitationsGrid";
import { LimitationsTakeawayBanner } from "@/components/research-insights/limitations/LimitationsTakeawayBanner";

// Future Work Tab Components
import { FutureWorkHeaderRow } from "@/components/research-insights/future-work/FutureWorkHeaderRow";
import { FutureWorkDirectionsGrid } from "@/components/research-insights/future-work/FutureWorkDirectionsGrid";
import { RoadmapAndImpactRow } from "@/components/research-insights/future-work/RoadmapAndImpactRow";
import { FutureWorkTakeawayBanner } from "@/components/research-insights/future-work/FutureWorkTakeawayBanner";

export default function ResearchInsightsPage() {
  const [activeTab, setActiveTab] = useState<ResearchTabId>("overview");

  return (
    <div className="space-y-5">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl sm:text-[26px] font-extrabold text-[#111613] tracking-tight">
          Research Insights
        </h1>
        <p className="text-[13px] text-[#555E57] mt-1">
          Key findings, analysis, and research outcomes from the loan default prediction project.
        </p>
      </div>

      {/* Row 1: 5 Tab Pills */}
      <ResearchTabsRow activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Tab 1: Overview */}
      {activeTab === "overview" && (
        <div className="space-y-5">
          {/* Key Findings Highlight Card (Full Width) */}
          <KeyFindingsRow />

          {/* Research Overview (50%) + Our Contributions (50%) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
            <ResearchOverviewCard />
            <OurContributionsCard />
          </div>

          {/* Model Comparison (50%) + Read Research (50%) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
            <ModelComparisonResearchCard />
            <ReadResearchCard />
          </div>
        </div>
      )}

      {/* Tab 2: Methodology (Exact match to uploaded design) */}
      {activeTab === "methodology" && (
        <div className="space-y-5">
          {/* Row 2: Methodology Overview Flowchart Card (Full Width) */}
          <MethodologyOverviewFlowCard />

          {/* Row 3: 3 Cards Across (Modeling Approach, Dataset & Setup, Evaluation Metrics) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
            <ModelingApproachCard />
            <DatasetSetupCard />
            <EvaluationMetricsCard />
          </div>

          {/* Row 4: Key Design Choices (Full Width) */}
          <KeyDesignChoicesCard />
        </div>
      )}

      {/* Tab 3: Findings (Exact match to uploaded design media_1793740660363_26bc6a2e.png) */}
      {activeTab === "findings" && (
        <div className="space-y-5">
          {/* Row 1: 4 KPI Cards */}
          <FindingsKpiRow />

          {/* Row 2: Model Comparison (Test Set) + ROC Curve */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
            <ModelComparisonTestCard />
            <MultiRocCurveCard />
          </div>

          {/* Row 3: Feature Importance (SHAP) + Key Findings */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
            <FeatureImportanceShapCard />
            <FindingsKeyInsightsCard />
          </div>
        </div>
      )}

      {/* Tab 4: Limitations (Exact match to uploaded design media_1793741273146_a73ff2ec.jpg) */}
      {activeTab === "limitations" && (
        <div className="space-y-5">
          {/* Row 1: Overview (60%) + In Short (40%) */}
          <LimitationsHeaderRow />

          {/* Rows 2 & 3: 6 Limitation Cards in 3x2 Grid */}
          <LimitationsGrid />

          {/* Row 4: Takeaway Banner (Full Width) */}
          <LimitationsTakeawayBanner />
        </div>
      )}

      {/* Tab 5: Future Work (Exact match to uploaded design media_1793741817703_fad5a67f.jpg) */}
      {activeTab === "future_work" && (
        <div className="space-y-5">
          {/* Row 1: Overview (60%) + In Short (40%) */}
          <FutureWorkHeaderRow />

          {/* Row 2: Key Future Work Directions (Section Header + 6 Cards) */}
          <FutureWorkDirectionsGrid />

          {/* Row 3: Proposed Roadmap (60%) + Expected Impact (40%) */}
          <RoadmapAndImpactRow />

          {/* Row 4: Takeaway Banner (Full Width) */}
          <FutureWorkTakeawayBanner />
        </div>
      )}
    </div>
  );
}
