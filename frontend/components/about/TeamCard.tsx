"use client";

import React from "react";
import { Users } from "lucide-react";

interface TeamMember {
  initials: string;
  name: string;
  role: string;
}

const MEMBERS: TeamMember[] = [
  {
    initials: "TS",
    name: "Tanmay Singh",
    role: "Project Lead & Full-Stack Developer",
  },
  {
    initials: "AM",
    name: "Ayush Mishra",
    role: "Data Pipeline & Model Development",
  },
  {
    initials: "RS",
    name: "Rohit Sharma",
    role: "Research & Evaluation",
  },
];

export function TeamCard() {
  return (
    <article className="cl-card p-5 sm:p-6 bg-white flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Users className="w-5 h-5 text-[#158444] shrink-0 stroke-[2.2]" />
        <h2 className="text-[15px] sm:text-[16px] font-bold text-[#111613] tracking-tight">
          Team
        </h2>
      </div>

      {/* Members List */}
      <div className="space-y-3.5">
        {MEMBERS.map((m) => (
          <div key={m.name} className="flex items-center gap-3">
            {/* Initials Avatar */}
            <div className="w-9 h-9 rounded-full bg-[#EAF5EE] text-[#158444] font-bold text-[12px] flex items-center justify-center shrink-0 shadow-2xs">
              {m.initials}
            </div>

            {/* Member Details */}
            <div className="min-w-0">
              <p className="text-[13px] font-bold text-[#111613] tracking-tight truncate leading-tight">
                {m.name}
              </p>
              <p className="text-[11.5px] text-[#555E57] mt-0.5 truncate leading-tight">
                {m.role}
              </p>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}
