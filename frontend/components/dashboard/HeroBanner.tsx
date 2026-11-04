import Link from "next/link";
import { BookOpen, Check, Play } from "lucide-react";
import { formatInt } from "@/lib/format";

const FEATURES = [
  "Analyze Applicant Data",
  "Predict Default Risk",
  "Explain the Result",
  "Explore Key Insights",
];

export function HeroBanner({ totalApplicants }: { totalApplicants?: number }) {
  const count = totalApplicants ? `${formatInt(Math.floor(totalApplicants / 1000))}K+` : "300K+";
  return (
    <section className="cl-card relative overflow-hidden p-6 lg:p-7 flex flex-col xl:flex-row items-center justify-between gap-6 bg-white">
      {/* Left text column */}
      <div className="flex-1 z-10 w-full">
        <h1 className="text-2xl sm:text-[30px] font-extrabold text-ink tracking-tight leading-tight">
          Welcome to <span className="text-[#158444]">CreditLens</span>
        </h1>
        <h2 className="text-[14px] sm:text-[15px] font-bold text-[#1F2A22] mt-1.5">
          A Big Data–Driven Loan Default Prediction System
        </h2>
        <p className="text-[12.5px] text-[#555E57] mt-2 max-w-[500px] leading-relaxed">
          Predict the probability of loan default, understand key risk factors, and explore insights from {count} loan applications using
          machine learning and big data.
        </p>
        <div className="flex flex-wrap items-center gap-3 mt-5">
          <Link
            href="/check-risk"
            className="inline-flex items-center gap-2 bg-[#158444] hover:bg-[#0F6B36] text-white text-[13px] font-semibold px-5 py-2.5 rounded-full shadow-xs transition-colors"
          >
            <Play className="w-3.5 h-3.5 fill-white text-white stroke-none" />
            <span>Check Loan Risk</span>
          </Link>
          <Link
            href="/about"
            className="inline-flex items-center gap-2 bg-white hover:bg-neutral-50 text-[#1B271F] border border-[#D1D5DB] text-[13px] font-semibold px-5 py-2.5 rounded-full transition-colors shadow-2xs"
          >
            <BookOpen className="w-4 h-4 text-[#158444] stroke-[2.2]" />
            <span>How It Works</span>
          </Link>
        </div>
      </div>

      {/* Center 3D Graphic + Right Checklist */}
      <div className="flex flex-col sm:flex-row items-center gap-6 xl:gap-8 z-10 shrink-0">
        <div className="w-[250px] h-[142px] relative shrink-0 select-none flex items-center justify-center">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/hero-3d-asset.png" alt="CreditLens 3D illustration" className="w-full h-full object-contain" />
        </div>

        {/* Clean checklist without enclosing box, matching reference image */}
        <div className="flex flex-col gap-3 shrink-0">
          {FEATURES.map((feat) => (
            <div key={feat} className="flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-full bg-[#D6F3DF] flex items-center justify-center shrink-0">
                <Check className="w-3 h-3 text-[#158444] stroke-[3]" />
              </span>
              <span className="text-[13px] font-semibold text-[#1A251D]">{feat}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
