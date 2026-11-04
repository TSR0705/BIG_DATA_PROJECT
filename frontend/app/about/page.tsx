"use client";

import React from "react";
import { ProjectOverviewCard } from "@/components/about/ProjectOverviewCard";
import { KeyHighlightsCard } from "@/components/about/KeyHighlightsCard";
import { ObjectivesCard } from "@/components/about/ObjectivesCard";
import { SystemArchitectureCard } from "@/components/about/SystemArchitectureCard";
import { TechStackCard } from "@/components/about/TechStackCard";
import { DatasetCard } from "@/components/about/DatasetCard";
import { TeamCard } from "@/components/about/TeamCard";
import { ProjectOutcomeBanner } from "@/components/about/ProjectOutcomeBanner";

export default function AboutPage() {
  return (
    <div className="space-y-5">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl sm:text-[26px] font-extrabold text-[#111613] tracking-tight">
          About the Project
        </h1>
        <p className="text-[13px] text-[#555E57] mt-1">
          An end-to-end, explainable loan default prediction system using Big Data and machine learning.
        </p>
      </div>

      {/* Row 1: Project Overview (~42%) + Key Highlights (~58%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 items-stretch">
        <div className="lg:col-span-5 flex flex-col">
          <ProjectOverviewCard />
        </div>
        <div className="lg:col-span-7 flex flex-col">
          <KeyHighlightsCard />
        </div>
      </div>

      {/* Row 2: Objectives (~33%) + System Architecture (~67%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 items-stretch">
        <div className="lg:col-span-4 flex flex-col">
          <ObjectivesCard />
        </div>
        <div className="lg:col-span-8 flex flex-col">
          <SystemArchitectureCard />
        </div>
      </div>

      {/* Row 3: Tech Stack + Dataset + Team (3-column grid) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 items-stretch">
        <TechStackCard />
        <DatasetCard />
        <TeamCard />
      </div>

      {/* Row 4: Project Outcome Banner (Full Width) */}
      <ProjectOutcomeBanner />
    </div>
  );
}
