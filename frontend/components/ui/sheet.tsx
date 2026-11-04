"use client";

import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;
export const SheetClose = DialogPrimitive.Close;

export function SheetContent({
  className,
  children,
  title,
  ...props
}: React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & { title: string }) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-ink/30 backdrop-blur-[2px]" />
      <DialogPrimitive.Content
        className={cn("fixed inset-y-0 left-0 z-50 w-[280px] max-w-[85vw] bg-white shadow-[var(--shadow-float)] outline-none", className)}
        {...props}
      >
        <DialogPrimitive.Title className="sr-only">{title}</DialogPrimitive.Title>
        <DialogPrimitive.Description className="sr-only">Navigation</DialogPrimitive.Description>
        <DialogPrimitive.Close className="absolute right-3 top-3 w-8 h-8 rounded-full flex items-center justify-center text-ink-muted hover:bg-app" aria-label="Close menu">
          <X className="w-4 h-4" />
        </DialogPrimitive.Close>
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}
