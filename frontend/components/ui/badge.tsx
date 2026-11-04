import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold whitespace-nowrap", {
  variants: {
    tone: {
      low: "bg-mint-soft text-forest",
      medium: "bg-warning-soft text-warning",
      high: "bg-danger-soft text-danger",
      neutral: "bg-app text-ink-muted",
    },
  },
  defaultVariants: { tone: "neutral" },
});

export function Badge({ className, tone, ...props }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

export function riskTone(tier: string): "low" | "medium" | "high" | "neutral" {
  const t = tier.toLowerCase();
  if (t.startsWith("low")) return "low";
  if (t.startsWith("med")) return "medium";
  if (t.startsWith("high")) return "high";
  return "neutral";
}
