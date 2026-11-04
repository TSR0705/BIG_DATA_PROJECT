"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Bell, ChevronDown, HelpCircle, Loader2, Menu, Search, Sparkles } from "lucide-react";
import { apiDocsUrl, ApiError, fetchRecentPredictions, scoreApplicant, searchApplicants } from "@/lib/api";
import { timeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ApplicantScore, ApplicantSearchHit, RecentPredictionData } from "@/types";
import { Badge, riskTone } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { SidebarNav } from "./SidebarNav";

const USER_NAME = process.env.NEXT_PUBLIC_USER_NAME || "Rahul Sharma";
const USER_ROLE = process.env.NEXT_PUBLIC_USER_ROLE || "Student";

const TIER_LABEL: Record<string, string> = { LOW: "Low Risk", MEDIUM: "Medium Risk", HIGH: "High Risk" };

/* ------------------------------------------------------------------ search */
function ApplicantSearch() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<ApplicantSearchHit[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scored, setScored] = useState<Record<number, ApplicantScore | "loading">>({});
  const inputRef = useRef<HTMLInputElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  // ⌘K / Ctrl+K focuses the search box
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // close on outside click
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // debounced ID search against the API
  useEffect(() => {
    const q = query.trim();
    if (!q) return;
    const handle = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await searchApplicants(q);
        setHits(res.results);
        setOpen(true);
      } catch (err) {
        setHits([]);
        setError(err instanceof ApiError ? err.message : "Search failed");
        setOpen(true);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  const score = useCallback(async (hit: ApplicantSearchHit) => {
    setScored((s) => ({ ...s, [hit.sk_id_curr]: "loading" }));
    try {
      const result = await scoreApplicant(hit.sk_id_curr);
      setScored((s) => ({ ...s, [hit.sk_id_curr]: result }));
    } catch (err) {
      setScored((s) => {
        const next = { ...s };
        delete next[hit.sk_id_curr];
        return next;
      });
      setError(err instanceof ApiError ? err.message : "Scoring failed");
    }
  }, []);

  const digitsOnly = query.trim().replace(/^app-?/i, "");
  const isIdQuery = digitsOnly === "" || /^\d+$/.test(digitsOnly);

  return (
    <div ref={boxRef} className="relative flex-1 max-w-xl">
      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-faint stroke-[2.2]" />
      <input
        ref={inputRef}
        type="search"
        role="combobox"
        aria-expanded={open}
        aria-controls="applicant-search-results"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          if (!e.target.value.trim()) {
            setHits([]);
            setError(null);
            setOpen(false);
          }
        }}
        onFocus={() => query && setOpen(true)}
        placeholder="Search for applicants, metrics, or insights..."
        className="w-full bg-white border border-line text-xs font-medium text-ink placeholder-ink-faint rounded-full pl-11 pr-16 py-2.5 focus:outline-none focus:ring-2 focus:ring-forest/15 focus:border-forest transition-all shadow-2xs"
      />
      <div className="absolute right-4 top-1/2 -translate-y-1/2 hidden sm:flex items-center gap-1 px-2 py-0.5 rounded-md bg-app border border-line-strong text-[10px] font-bold text-ink-faint pointer-events-none">
        <span>⌘</span>
        <span>K</span>
      </div>

      {open && (
        <div
          id="applicant-search-results"
          role="listbox"
          className="absolute left-0 right-0 top-[calc(100%+8px)] z-40 rounded-2xl border border-line bg-white shadow-[var(--shadow-float)] p-2 max-h-[22rem] overflow-y-auto"
        >
          {loading && (
            <div className="flex items-center gap-2 px-3 py-2 text-xs text-ink-faint">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Searching applicant IDs…
            </div>
          )}
          {!loading && error && <p className="px-3 py-2 text-xs text-danger">{error}</p>}
          {!loading && !error && !isIdQuery && (
            <p className="px-3 py-2 text-xs text-ink-muted">
              Search works on applicant IDs (SK_ID_CURR), e.g. <span className="font-mono">100002</span>. Metrics and insights are on the
              <Link href="/model-performance" className="text-forest font-semibold"> Model Performance</Link> and
              <Link href="/research-insights" className="text-forest font-semibold"> Research Insights</Link> pages.
            </p>
          )}
          {!loading && !error && isIdQuery && hits.length === 0 && query.trim() && (
            <p className="px-3 py-2 text-xs text-ink-muted">No applicant ID starts with “{digitsOnly}”.</p>
          )}
          {hits.map((hit) => {
            const s = scored[hit.sk_id_curr];
            return (
              <div key={hit.sk_id_curr} role="option" aria-selected={false} className="flex items-center justify-between gap-3 px-3 py-2 rounded-xl hover:bg-mint-soft/60">
                <div className="min-w-0">
                  <p className="text-xs font-bold text-ink">{hit.applicant_id}</p>
                  <p className="text-[10.5px] text-ink-faint">{hit.source === "labelled" ? "Labelled application" : "Inference-only application"}</p>
                </div>
                {s === "loading" ? (
                  <Loader2 className="w-4 h-4 animate-spin text-forest" />
                ) : s ? (
                  <div className="flex items-center gap-2">
                    <Badge tone={riskTone(TIER_LABEL[s.risk_tier] ?? s.risk_tier)}>{TIER_LABEL[s.risk_tier] ?? s.risk_tier}</Badge>
                    <span className="text-xs font-bold text-ink tabular-nums">{s.default_percentage.toFixed(1)}%</span>
                  </div>
                ) : (
                  <Button size="sm" variant="pill" onClick={() => score(hit)}>
                    <Sparkles className="w-3 h-3" /> Score
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ notifications */
function Notifications() {
  const [items, setItems] = useState<RecentPredictionData[] | null>(null);
  const [fresh, setFresh] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    fetchRecentPredictions(5)
      .then((d) => {
        setItems(d);
        setFresh(d.some((p) => Date.now() - new Date(p.scored_at).getTime() < 24 * 3600 * 1000));
        setError(null);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Unavailable"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <DropdownMenu onOpenChange={(o) => o && load()}>
      <DropdownMenuTrigger asChild>
        <Button size="icon" variant="ghost" aria-label="Notifications" className="relative">
          <Bell className="w-4 h-4 stroke-[2]" />
          {fresh && <span className="absolute top-2.5 right-2.5 w-2 h-2 rounded-full bg-emerald ring-2 ring-white" />}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        <DropdownMenuLabel>Latest model predictions</DropdownMenuLabel>
        {error && <p className="px-3 py-2 text-xs text-danger">{error}</p>}
        {items && items.length === 0 && <p className="px-3 py-2 text-xs text-ink-muted">No predictions logged yet.</p>}
        {(items ?? []).map((p) => (
          <DropdownMenuItem key={`${p.applicant_id}-${p.scored_at}`} className="justify-between">
            <span className="flex flex-col">
              <span className="font-bold text-ink">{p.applicant_id}</span>
              <span className="text-[10.5px] text-ink-faint">{timeAgo(p.scored_at)}</span>
            </span>
            <span className="flex items-center gap-2">
              <Badge tone={riskTone(p.risk_tier)}>{p.risk_tier}</Badge>
              <span className="tabular-nums font-bold">{p.probability_percent}</span>
            </span>
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link href="/understand-results">View all predictions</Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/* ------------------------------------------------------------------ profile */
function Avatar() {
  return (
    <div className="w-9 h-9 rounded-full overflow-hidden border border-[#D5DDD7] flex items-center justify-center shadow-2xs shrink-0">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/avatar_rahul.png" alt="Rahul Sharma" className="w-full h-full object-cover" />
    </div>
  );
}

function ProfileMenu() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" className="flex items-center gap-3 pl-2 py-1 rounded-full hover:bg-white/70 transition-colors select-none" aria-label="Account menu">
          <Avatar />
          <div className="hidden sm:block text-left leading-tight">
            <p className="text-xs font-bold text-ink">{USER_NAME}</p>
            <p className="text-[10px] text-ink-faint font-medium">{USER_ROLE}</p>
          </div>
          <ChevronDown className="w-3.5 h-3.5 text-ink-faint stroke-[2.2]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          {USER_NAME} · {USER_ROLE}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link href="/about">About the project</Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <a href={apiDocsUrl()} target="_blank" rel="noreferrer">
            API documentation
          </a>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <a href="https://github.com/TSR0705/BIG_DATA_PROJECT" target="_blank" rel="noreferrer">
            Source repository
          </a>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/* ------------------------------------------------------------------ header */
export function Header() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  return (
    <header className="h-16 px-4 sm:px-6 flex items-center justify-between gap-3 sm:gap-6 shrink-0 sticky top-0 z-20 bg-app/90 backdrop-blur-md">
      <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
        <SheetTrigger asChild>
          <Button size="icon" variant="ghost" className="lg:hidden" aria-label="Open navigation">
            <Menu className="w-4 h-4 stroke-[2.2]" />
          </Button>
        </SheetTrigger>
        <SheetContent title="Navigation">
          <SidebarNav onNavigate={() => setDrawerOpen(false)} />
        </SheetContent>
      </Sheet>

      <ApplicantSearch />

      <div className={cn("flex items-center gap-2 sm:gap-3")}>
        <Notifications />
        <Link href="/about" aria-label="Help and documentation" className={cn(buttonVariants({ size: "icon", variant: "ghost" }))}>
          <HelpCircle className="w-4 h-4 stroke-[2]" />
        </Link>
        <ProfileMenu />
      </div>
    </header>
  );
}
