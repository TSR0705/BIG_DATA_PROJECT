import { Home, User, BarChart3, Layers, Database, FlaskConical, Info, type LucideIcon } from "lucide-react";

export interface NavItem {
  name: string;
  href: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavItem[] = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Check Loan Risk", href: "/check-risk", icon: User },
  { name: "Understand Results", href: "/understand-results", icon: BarChart3 },
  { name: "Model Performance", href: "/model-performance", icon: Layers },
  { name: "Data & Pipeline", href: "/data-pipeline", icon: Database },
  { name: "Research Insights", href: "/research-insights", icon: FlaskConical },
];

export const SECONDARY_NAV: NavItem[] = [{ name: "About the Project", href: "/about", icon: Info }];
