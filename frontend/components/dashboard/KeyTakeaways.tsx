import { Check, Lightbulb } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";

export function KeyTakeaways({ takeaways }: { takeaways: string[] }) {
  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="items-center">
        <CardTitle icon={<Lightbulb />} iconClass="bg-[#FEF3C7] text-[#D97706]">
          Key Takeaways
        </CardTitle>
      </CardHeader>
      <ul className="mt-3.5 space-y-3">
        {takeaways.map((t) => (
          <li key={t} className="flex items-start gap-2.5">
            <span className="w-4 h-4 rounded-full bg-[#E0F9E6] text-[#158444] border border-[#A6F4C5]/60 flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
              <Check className="w-2.5 h-2.5 stroke-[3]" />
            </span>
            <span className="text-[12.5px] text-[#1F2923] leading-snug font-medium">{t}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
