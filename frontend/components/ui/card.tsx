import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("cl-card p-5", className)} {...props} />;
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex items-start justify-between gap-3", className)} {...props} />;
}

export function CardTitle({
  icon,
  children,
  subtitle,
  className,
}: {
  icon?: React.ReactNode;
  children: React.ReactNode;
  subtitle?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start gap-2.5 min-w-0", className)}>
      {icon ? (
        <div className="w-8 h-8 rounded-lg bg-mint-soft text-forest flex items-center justify-center shrink-0 mt-0.5 [&_svg]:w-4 [&_svg]:h-4 [&_svg]:stroke-[2.2]">
          {icon}
        </div>
      ) : null}
      <div className="min-w-0">
        <h3 className="text-[15px] font-bold text-ink leading-tight truncate">{children}</h3>
        {subtitle ? <p className="text-[11.5px] text-ink-faint mt-0.5 truncate">{subtitle}</p> : null}
      </div>
    </div>
  );
}
