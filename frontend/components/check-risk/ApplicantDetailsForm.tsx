"use client";

import React, { useState } from "react";
import { Briefcase, Calendar, ChevronDown, Coins, Play, RefreshCw, User } from "lucide-react";
import type { PredictRiskInput } from "@/types";

interface Props {
  initialValues: PredictRiskInput;
  onSubmit: (values: PredictRiskInput) => void;
  onLoadExample: () => void;
  loading: boolean;
}

export function ApplicantDetailsForm({
  initialValues,
  onSubmit,
  onLoadExample,
  loading,
}: Props) {
  const [age, setAge] = useState<number>(initialValues.age);
  const [annualIncome, setAnnualIncome] = useState<string>(() =>
    Number(initialValues.annual_income).toLocaleString("en-IN")
  );
  const [employmentType, setEmploymentType] = useState<string>(initialValues.employment_type);
  const [loanAmount, setLoanAmount] = useState<string>(() =>
    Number(initialValues.loan_amount).toLocaleString("en-IN")
  );
  const [loanTenure, setLoanTenure] = useState<number>(initialValues.loan_tenure_months);
  const [existingObligations, setExistingObligations] = useState<string>(() =>
    Number(initialValues.existing_obligations).toLocaleString("en-IN")
  );

  function handleExampleClick() {
    setAge(32);
    setAnnualIncome((850000).toLocaleString("en-IN"));
    setEmploymentType("Salaried");
    setLoanAmount((1200000).toLocaleString("en-IN"));
    setLoanTenure(60);
    setExistingObligations((18000).toLocaleString("en-IN"));
    onLoadExample();
  }

  function parseFormatted(val: string): number {
    return parseFloat(val.replace(/,/g, "")) || 0;
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      age,
      annual_income: parseFormatted(annualIncome),
      employment_type: employmentType,
      loan_amount: parseFormatted(loanAmount),
      loan_tenure_months: loanTenure,
      existing_obligations: parseFormatted(existingObligations),
    });
  }

  return (
    <article className="cl-card p-5 lg:p-6 flex flex-col justify-between h-full bg-white">
      <div>
        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-2">
          <div className="flex items-start gap-3">
            <span className="w-10 h-10 rounded-xl bg-[#EAF5EE] text-[#158444] flex items-center justify-center shrink-0 shadow-2xs mt-0.5">
              <User className="w-5 h-5 stroke-[2.2]" />
            </span>
            <div>
              <h2 className="text-[17px] font-bold text-[#111613] leading-snug">Applicant Details</h2>
              <p className="text-[12px] text-[#89938D] mt-0.5">
                Provide basic financial and personal information about the applicant.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={handleExampleClick}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-[#D5DDD7] bg-white hover:bg-[#F8FAF8] text-[12px] font-semibold text-[#111613] transition-colors shadow-2xs shrink-0 cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[#158444] stroke-[2.4]" />
            <span>Load Example</span>
          </button>
        </div>

        {/* 2-Column Form Fields */}
        <form id="check-risk-form" onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-3.5 mt-4">
          {/* Age */}
          <div>
            <label className="block text-[12.5px] font-semibold text-[#111613] mb-1.5">
              Age
            </label>
            <input
              type="number"
              min={18}
              max={100}
              value={age}
              onChange={(e) => setAge(parseInt(e.target.value) || 18)}
              className="w-full h-11 px-3.5 rounded-xl border border-[#D5DDD7] text-[13.5px] font-medium text-[#111613] focus:outline-hidden focus:border-[#158444] focus:ring-1 focus:ring-[#158444] transition-all bg-white"
            />
          </div>

          {/* Annual Income */}
          <div>
            <label className="block text-[12.5px] font-semibold text-[#111613] mb-1.5">
              Annual Income (₹)
            </label>
            <input
              type="text"
              value={annualIncome}
              onChange={(e) => setAnnualIncome(e.target.value)}
              className="w-full h-11 px-3.5 rounded-xl border border-[#D5DDD7] text-[13.5px] font-medium text-[#111613] focus:outline-hidden focus:border-[#158444] focus:ring-1 focus:ring-[#158444] transition-all bg-white"
            />
          </div>

          {/* Employment Type */}
          <div>
            <label className="block text-[12.5px] font-semibold text-[#111613] mb-1.5">
              Employment Type
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#89938D]">
                <Briefcase className="w-4 h-4 stroke-[2]" />
              </span>
              <select
                value={employmentType}
                onChange={(e) => setEmploymentType(e.target.value)}
                className="w-full h-11 pl-10 pr-9 rounded-xl border border-[#D5DDD7] text-[13.5px] font-medium text-[#111613] focus:outline-hidden focus:border-[#158444] focus:ring-1 focus:ring-[#158444] transition-all bg-white appearance-none cursor-pointer"
              >
                <option value="Salaried">Salaried</option>
                <option value="Commercial associate">Commercial associate</option>
                <option value="State servant">State servant</option>
                <option value="Working">Working</option>
                <option value="Self-employed">Self-employed</option>
              </select>
              <span className="absolute right-3.5 top-1/2 -translate-y-1/2 pointer-events-none text-[#89938D]">
                <ChevronDown className="w-4 h-4 stroke-[2.2]" />
              </span>
            </div>
          </div>

          {/* Loan Amount */}
          <div>
            <label className="block text-[12.5px] font-semibold text-[#111613] mb-1.5">
              Loan Amount (₹)
            </label>
            <input
              type="text"
              value={loanAmount}
              onChange={(e) => setLoanAmount(e.target.value)}
              className="w-full h-11 px-3.5 rounded-xl border border-[#D5DDD7] text-[13.5px] font-medium text-[#111613] focus:outline-hidden focus:border-[#158444] focus:ring-1 focus:ring-[#158444] transition-all bg-white"
            />
          </div>

          {/* Loan Tenure */}
          <div>
            <label className="block text-[12.5px] font-semibold text-[#111613] mb-1.5">
              Loan Tenure (Months)
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#89938D]">
                <Calendar className="w-4 h-4 stroke-[2]" />
              </span>
              <input
                type="number"
                min={6}
                max={360}
                value={loanTenure}
                onChange={(e) => setLoanTenure(parseInt(e.target.value) || 12)}
                className="w-full h-11 pl-10 pr-3.5 rounded-xl border border-[#D5DDD7] text-[13.5px] font-medium text-[#111613] focus:outline-hidden focus:border-[#158444] focus:ring-1 focus:ring-[#158444] transition-all bg-white"
              />
            </div>
          </div>

          {/* Existing Obligations */}
          <div>
            <label className="block text-[12.5px] font-semibold text-[#111613] mb-1.5">
              Existing Obligations (₹/Month)
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#89938D]">
                <Coins className="w-4 h-4 stroke-[2]" />
              </span>
              <input
                type="text"
                value={existingObligations}
                onChange={(e) => setExistingObligations(e.target.value)}
                className="w-full h-11 pl-10 pr-3.5 rounded-xl border border-[#D5DDD7] text-[13.5px] font-medium text-[#111613] focus:outline-hidden focus:border-[#158444] focus:ring-1 focus:ring-[#158444] transition-all bg-white"
              />
            </div>
          </div>
        </form>
      </div>

      {/* Predict Button */}
      <div className="mt-5 pt-3">
        <button
          type="submit"
          form="check-risk-form"
          disabled={loading}
          className="w-full h-12 rounded-full bg-[#158444] hover:bg-[#0F6B36] text-white text-[14px] font-bold flex items-center justify-center gap-2 transition-all shadow-xs disabled:opacity-60 cursor-pointer"
        >
          <Play className="w-4 h-4 fill-white text-white" />
          <span>{loading ? "Calculating..." : "Predict Loan Default Risk"}</span>
        </button>
      </div>
    </article>
  );
}
