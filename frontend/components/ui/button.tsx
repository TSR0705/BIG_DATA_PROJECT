import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest/30 disabled:pointer-events-none disabled:opacity-50 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-forest text-white shadow-xs hover:bg-forest-dark",
        outline: "bg-white text-ink border border-forest hover:bg-app",
        ghost: "bg-transparent text-ink-muted hover:bg-app hover:text-ink",
        pill: "bg-white text-ink-muted border border-line hover:text-forest hover:border-emerald/30 shadow-2xs",
      },
      size: {
        sm: "h-7 px-3 text-[11px]",
        md: "h-9 px-5",
        icon: "h-10 w-10 rounded-full border border-line bg-white text-ink-muted hover:bg-app shadow-2xs",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, type = "button", ...props }, ref) => (
    <button ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  ),
);
Button.displayName = "Button";

export { buttonVariants };
