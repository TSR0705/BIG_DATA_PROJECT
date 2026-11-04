"use client";

import React, { useState } from "react";
import { BookOpen, ExternalLink, FileText, X, CheckCircle2 } from "lucide-react";

interface DocItem {
  id: string;
  title: string;
  subtitle: string;
  icon: React.ElementType;
  badgeBg: string;
  iconColor: string;
  badgeText: string;
  abstract: string;
  sections: { heading: string; content: string }[];
  pages: number;
}

const DOCS: DocItem[] = [
  {
    id: "ieee-paper",
    title: "Research Paper (IEEE Format)",
    subtitle: "Complete research and experimental analysis",
    icon: FileText,
    badgeBg: "bg-[#FDE8E8]",
    iconColor: "text-[#E02424]",
    badgeText: "IEEE Transactions / Research Paper",
    pages: 14,
    abstract:
      "This study presents CreditLens, an explainable loan default prediction framework using Apache Spark distributed data engineering and an isotonic-calibrated XGBoost classifier evaluated across 307,511 loan applications. On the independent Phase 11 out-of-sample benchmark (N=46,127), the model achieved ROC-AUC 0.7909, PR-AUC 0.2853, Brier Score 0.0654, and Expected Calibration Error (ECE) 0.0042, while TreeSHAP attribution uncovered loan amount and applicant age as dominant structural determinants of credit default.",
    sections: [
      {
        heading: "1. Problem Formulation & Big Data Pipeline",
        content:
          "Retail credit assessment faces severe class imbalance (8.07% baseline default rate) and high dimensional tabular data. An Apache Spark Medallion pipeline (Bronze → Silver → Gold) processed 58.5M relational entries into 240 engineered behavioral and domain features with strict leakage prevention.",
      },
      {
        heading: "2. Calibration & Monotonicity Protocol",
        content:
          "Standard gradient boosted trees output uncalibrated margins. An independent 5-fold cross-validated isotonic calibrator was fitted strictly on training out-of-fold predictions. Empirical testing verified monotonic risk separation: Low (<10%), Medium (10–30%), and High (>30%).",
      },
      {
        heading: "3. Explainability & Subgroup Robustness",
        content:
          "TreeSHAP stability experiments (E5) demonstrated high top-10 feature ranking agreement (Kendall's tau > 0.88) across bootstrap retraining iterations. Demographic audits (E7) confirmed calibration consistency across sex and age cohorts.",
      },
    ],
  },
  {
    id: "project-report",
    title: "Project Report",
    subtitle: "Methodology, implementation and results",
    icon: FileText,
    badgeBg: "bg-[#FEF3C7]",
    iconColor: "text-[#D97706]",
    badgeText: "Comprehensive Engineering Report",
    pages: 38,
    abstract:
      "Detailed architectural and engineering documentation covering the end-to-end lifecycle of the CreditLens loan default prediction system, including environment setup, Spark transformations, feature pipelines, baseline comparisons, hyperparameter tuning, model serialization, FastAPI inference, and Next.js frontend delivery.",
    sections: [
      {
        heading: "1. Data Engineering Architecture",
        content:
          "Spark DataFrame transformations joining application, bureau, previous application, POS cash balance, and installment payments tables. All aggregations respect temporal cutoff to eliminate target leakage.",
      },
      {
        heading: "2. Machine Learning Model Suite",
        content:
          "Benchmarked 4 primary algorithms: Logistic Regression (ROC-AUC 0.7482), Random Forest (0.7796), LightGBM (0.7881), and XGBoost (0.7909). Hyperparameter grid search tuned subsample ratio, learning rate, and tree depth.",
      },
      {
        heading: "3. API & Dashboard Integration",
        content:
          "Production inference service built with FastAPI, Pydantic input schemas, frozen model artifacts, and REST endpoints delivering real-time probability scores and localized SHAP factor contributions.",
      },
    ],
  },
  {
    id: "presentation-deck",
    title: "Presentation (PPT)",
    subtitle: "Project overview and key findings",
    icon: BookOpen,
    badgeBg: "bg-[#EAF5EE]",
    iconColor: "text-[#158444]",
    badgeText: "Executive Slide Deck",
    pages: 22,
    abstract:
      "Executive and technical presentation slides summarizing the CreditLens project motivation, Big Data architecture, model performance milestones, TreeSHAP explainability findings, and business deployment strategy for automated loan underwriting.",
    sections: [
      {
        heading: "Slide Highlights",
        content:
          "• Slide 1–4: Problem Statement, Credit Risk Dynamics & High-Dimensional Imbalance.\n• Slide 5–9: Spark Medallion Pipeline & Feature Engineering Taxonomy.\n• Slide 10–15: Benchmark Model Comparison (XGBoost vs. LightGBM, Random Forest, Logistic Regression).\n• Slide 16–19: Probability Calibration & TreeSHAP Attribution Insights.\n• Slide 20–22: Risk Stratification, Limitations, and Future Roadmap.",
      },
    ],
  },
];

export function ReadResearchCard() {
  const [selectedDoc, setSelectedDoc] = useState<DocItem | null>(null);

  return (
    <>
      <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
        <div>
          {/* Header */}
          <div className="flex items-start gap-3 mb-4">
            <FileText className="w-6 h-6 text-[#158444] stroke-[2.2] shrink-0 mt-0.5" />
            <div>
              <h2 className="text-[17px] font-bold text-[#111613] leading-snug">
                Read Research
              </h2>
              <p className="text-[11.5px] text-[#89938D] mt-0.5">
                Access the project report, research paper, and documentation.
              </p>
            </div>
          </div>

          {/* 3 Document Rows */}
          <div className="space-y-3.5 mt-4">
            {DOCS.map((doc) => {
              const Icon = doc.icon;
              return (
                <div
                  key={doc.title}
                  className="flex items-center justify-between gap-3 p-2.5 rounded-2xl hover:bg-[#FAFBF9] border border-transparent hover:border-[#EEF2EE] transition-all"
                >
                  {/* Left: Badge + Titles */}
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={`w-10 h-10 rounded-full ${doc.badgeBg} ${doc.iconColor} flex items-center justify-center shrink-0 shadow-2xs`}
                    >
                      <Icon className="w-5 h-5 stroke-[2]" />
                    </div>
                    <div className="min-w-0">
                      <h3 className="text-[13px] font-bold text-[#111613] leading-tight truncate">
                        {doc.title}
                      </h3>
                      <p className="text-[11.5px] text-[#555E57] mt-0.5 leading-tight truncate">
                        {doc.subtitle}
                      </p>
                    </div>
                  </div>

                  {/* Right: View Action Button */}
                  <button
                    type="button"
                    onClick={() => setSelectedDoc(doc)}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-[#D5DDD7] bg-white hover:bg-neutral-50 text-[12px] font-semibold text-[#111613] transition-colors shadow-2xs shrink-0 cursor-pointer"
                  >
                    <span>View</span>
                    <ExternalLink className="w-3.5 h-3.5 text-[#555E57] stroke-[2]" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </article>

      {/* Document Viewer Modal */}
      {selectedDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-3xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-[#EEF2EE] animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="p-6 border-b border-[#EEF2EE] flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <div
                  className={`w-11 h-11 rounded-2xl ${selectedDoc.badgeBg} ${selectedDoc.iconColor} flex items-center justify-center shrink-0`}
                >
                  <selectedDoc.icon className="w-6 h-6 stroke-[2]" />
                </div>
                <div>
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#EAF5EE] text-[#158444] text-[11px] font-bold mb-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>{selectedDoc.badgeText}</span>
                  </div>
                  <h3 className="text-[18px] font-extrabold text-[#111613] leading-tight">
                    {selectedDoc.title}
                  </h3>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelectedDoc(null)}
                className="w-9 h-9 rounded-full bg-[#F4F5F6] hover:bg-[#EAECEE] text-[#555E57] flex items-center justify-center transition-colors cursor-pointer shrink-0"
              >
                <X className="w-5 h-5 stroke-[2]" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-5 text-[#111613]">
              <div>
                <h4 className="text-[12px] font-bold text-[#89938D] uppercase tracking-wider mb-1.5">
                  Executive Abstract
                </h4>
                <p className="text-[13.5px] text-[#444D46] leading-relaxed bg-[#F8FAF8] p-4 rounded-2xl border border-[#EEF2EE]">
                  {selectedDoc.abstract}
                </p>
              </div>

              <div className="space-y-4">
                <h4 className="text-[12px] font-bold text-[#89938D] uppercase tracking-wider">
                  Document Content
                </h4>
                {selectedDoc.sections.map((sec, idx) => (
                  <div key={idx} className="space-y-1">
                    <h5 className="text-[14px] font-bold text-[#111613]">{sec.heading}</h5>
                    <p className="text-[13px] text-[#555E57] leading-relaxed whitespace-pre-line">
                      {sec.content}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 sm:p-5 border-t border-[#EEF2EE] bg-[#FAFBF9] rounded-b-3xl flex items-center justify-between">
              <span className="text-[12px] text-[#89938D]">
                Length: {selectedDoc.pages} pages • Phase 11 Verified
              </span>
              <button
                type="button"
                onClick={() => setSelectedDoc(null)}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#158444] hover:bg-[#106A36] text-white text-[12.5px] font-bold transition-colors cursor-pointer shadow-2xs"
              >
                <span>Close Document</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
