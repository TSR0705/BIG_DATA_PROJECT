import { SidebarNav } from "./SidebarNav";

/** Fixed desktop sidebar that stays strictly in place during page scroll. */
export function Sidebar() {
  return (
    <aside className="hidden lg:flex fixed top-0 bottom-0 left-0 w-64 bg-white border-r border-line z-30 flex-col overflow-y-auto">
      <SidebarNav />
    </aside>
  );
}
