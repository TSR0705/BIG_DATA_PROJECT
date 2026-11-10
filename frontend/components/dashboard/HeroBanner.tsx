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
  const count = totalApplicants ? `${formatInt(Math.floor(totalApplicants / 1000))}K+` : "307K+";

  return (
    <section className="cl-card relative overflow-hidden p-6 sm:p-7 xl:p-8 bg-white border border-[#E7ECE9] rounded-2xl sm:rounded-3xl shadow-2xs">
      <div className="flex flex-col lg:flex-row items-center justify-between gap-6 lg:gap-4 xl:gap-8">
        
        {/* Left text & CTA region (~42-45% width on desktop) */}
        <div className="flex-1 w-full lg:max-w-[46%] z-10">
          <h1 className="text-2xl sm:text-3xl lg:text-[34px] xl:text-[36px] font-extrabold text-[#101828] tracking-tight leading-[1.15]">
            Welcome to <span className="text-[#087A43]">CreditLens</span>
          </h1>
          <h2 className="text-[14.5px] sm:text-[15.5px] font-bold text-[#101828] mt-2 tracking-tight">
            A Big Data–Driven Loan Default Prediction System
          </h2>
          <p className="text-[12.5px] sm:text-[13px] text-[#475467] mt-2.5 max-w-[520px] leading-relaxed">
            Predict the probability of loan default, understand key risk factors, and explore insights from {count} loan applications using
            machine learning and big data.
          </p>

          <div className="flex flex-wrap items-center gap-3.5 mt-5 sm:mt-6">
            <Link
              href="/check-risk"
              className="inline-flex items-center gap-2 bg-[#087A43] hover:bg-[#066336] text-white text-[13px] sm:text-[13.5px] font-semibold px-5.5 sm:px-6 py-2.5 rounded-full shadow-xs hover:shadow-sm transition-all"
            >
              <Play className="w-3.5 h-3.5 fill-none stroke-white stroke-[2.4]" />
              <span>Check Loan Risk</span>
            </Link>
            <Link
              href="/about"
              className="inline-flex items-center gap-2 bg-white hover:bg-[#F2FAF4] text-[#101828] border-2 border-[#087A43]/70 hover:border-[#087A43] text-[13px] sm:text-[13.5px] font-semibold px-5 sm:px-5.5 py-2.5 rounded-full transition-all shadow-2xs"
            >
              <BookOpen className="w-4 h-4 text-[#087A43] stroke-[2.2]" />
              <span>How It Works</span>
            </Link>
          </div>
        </div>

        {/* Center 3D Illustration region (~35-38% width on desktop) */}
        <div className="relative w-full sm:w-[320px] md:w-[350px] lg:w-[340px] xl:w-[380px] 2xl:w-[420px] h-[170px] sm:h-[185px] lg:h-[190px] xl:h-[200px] 2xl:h-[220px] flex items-center justify-center shrink-0 z-10 select-none">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/hero-3d-asset.png"
            alt="CreditLens 3D Loan Default Prediction Illustration"
            className="w-full h-full object-contain"
          />
        </div>

        {/* Right Feature Checklist Card region (~20-22% width on desktop) */}
        <div className="w-full lg:w-auto shrink-0 z-10">
          <div className="bg-white border border-[#E7ECE9] rounded-2xl p-4 sm:p-4.5 shadow-2xs flex flex-col gap-3 min-w-[210px] xl:min-w-[220px]">
            {FEATURES.map((feat) => (
              <div key={feat} className="flex items-center gap-2.5">
                <span className="w-5 h-5 rounded-full bg-[#D6F3DF] flex items-center justify-center shrink-0">
                  <Check className="w-3 h-3 text-[#087A43] stroke-[3]" />
                </span>
                <span className="text-[12.5px] sm:text-[13px] font-bold text-[#101828] whitespace-nowrap">
                  {feat}
                </span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </section>
  );
}
