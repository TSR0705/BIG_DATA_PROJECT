"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { ApiError, API_BASE_URL, fetchProjectOverview } from "@/lib/api";
import type { ProjectOverviewData } from "@/types";
import { Button } from "@/components/ui/button";
import { HeroBanner } from "@/components/dashboard/HeroBanner";
import { KPICards } from "@/components/dashboard/KPICards";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { DefaultRateProfile } from "@/components/dashboard/DefaultRateProfile";
import { ApplicantDistribution } from "@/components/dashboard/ApplicantDistribution";
import { RecentPredictions } from "@/components/dashboard/RecentPredictions";
import { TopRiskFactors } from "@/components/dashboard/TopRiskFactors";
import { KeyTakeaways } from "@/components/dashboard/KeyTakeaways";
import { DashboardSkeleton } from "@/components/dashboard/DashboardSkeleton";

type State = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; data: ProjectOverviewData };

export default function DashboardPage() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetchProjectOverview("age")
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((err) => {
        if (!cancelled) setState({ status: "error", message: err instanceof ApiError ? err.message : "Unexpected error" });
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const load = useCallback(() => {
    setState({ status: "loading" });
    setAttempt((n) => n + 1);
  }, []);

  if (state.status === "loading") {
    return (
      <div className="space-y-4">
        <HeroBanner />
        <DashboardSkeleton />
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="space-y-4">
        <HeroBanner />
        <div role="alert" className="cl-card p-6 flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <span className="w-10 h-10 rounded-full bg-danger-soft text-danger flex items-center justify-center shrink-0">
            <AlertCircle className="w-5 h-5" />
          </span>
          <div className="flex-1">
            <h3 className="text-sm font-bold text-ink">Dashboard data unavailable</h3>
            <p className="text-xs text-ink-muted mt-1">
              {state.message}. Start the API with <code className="font-mono bg-app px-1 rounded">uvicorn api.app.main:app --port 8000</code> from the
              project root (expected at {API_BASE_URL}). Nothing is shown from placeholder data.
            </p>
          </div>
          <Button variant="outline" onClick={load}>
            <RefreshCw className="w-3.5 h-3.5" /> Retry
          </Button>
        </div>
      </div>
    );
  }

  const { data } = state;
  return (
    <div className="space-y-4">
      <HeroBanner totalApplicants={data.total_applicants} />

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        {/* Main Left Section (8 cols on desktop) */}
        <div className="xl:col-span-8 flex flex-col gap-4">
          {/* Row 1: Top 3 KPI Cards */}
          <KPICards kpis={data.kpis} />

          {/* Row 2: Loan Default Trend + Applicant Distribution */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
            <div className="md:col-span-7">
              <DefaultRateProfile key={data.generated_at} initial={data.default_rate_profile} dimensions={data.profile_dimensions} />
            </div>
            <div className="md:col-span-5">
              <ApplicantDistribution distribution={data.distribution} />
            </div>
          </div>

          {/* Row 3: Top Risk Factors + Key Takeaways */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <TopRiskFactors factors={data.top_risk_factors} context={data.shap_context} />
            <KeyTakeaways takeaways={data.key_takeaways} />
          </div>
        </div>

        {/* Dedicated Right Section (4 cols on desktop; 2-col side-by-side on tablet) */}
        <div className="xl:col-span-4 grid grid-cols-1 md:grid-cols-2 xl:flex xl:flex-col gap-4">
          <QuickActions />
          <RecentPredictions predictions={data.recent_predictions} />
        </div>
      </div>

      {data.data_sources && (
        <p className="text-[10.5px] text-ink-faint px-1">
          Data sources: {Object.values(data.data_sources).join(" · ")}. Model: {data.model_version || "XGBoost (Phase 8) + Isotonic OOF (Phase 11)"}.
        </p>
      )}
    </div>
  );
}
