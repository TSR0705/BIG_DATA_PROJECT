import { CheckCircle2, Lightbulb } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";

export function KeyTakeaways({ takeaways }: { takeaways: string[] }) {
  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="items-center">
        <CardTitle icon={<Lightbulb />}>Key Takeaways</CardTitle>
      </CardHeader>
      <ul className="mt-3 space-y-2.5">
        {takeaways.map((t) => (
          <li key={t} className="flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 fill-[#158444] text-white shrink-0 mt-0.5" />
            <span className="text-[12.5px] text-ink leading-snug">{t}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
