import { Skeleton } from "@/components/ui/skeleton";

export function DashboardSkeleton() {
  return (
    <div className="space-y-4" aria-busy="true" aria-label="Loading dashboard">
      <Skeleton className="h-[190px] rounded-[22px]" />
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        <div className="xl:col-span-3 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Skeleton className="h-[150px] rounded-[22px]" />
          <Skeleton className="h-[150px] rounded-[22px]" />
          <Skeleton className="h-[150px] rounded-[22px]" />
        </div>
        <Skeleton className="h-[150px] rounded-[22px]" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <Skeleton className="lg:col-span-5 h-[260px] rounded-[22px]" />
        <Skeleton className="lg:col-span-4 h-[260px] rounded-[22px]" />
        <Skeleton className="lg:col-span-3 h-[260px] rounded-[22px]" />
      </div>
    </div>
  );
}
