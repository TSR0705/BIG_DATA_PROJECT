import type { Metadata, Viewport } from "next";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import "./globals.css";

export const metadata: Metadata = {
  title: "CreditLens — Loan Default Prediction",
  description: "A Big Data–Driven Loan Default Prediction System using Apache Spark and Explainable AI",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full bg-app text-ink antialiased">
        <Sidebar />
        <div className="lg:pl-64 flex flex-col min-w-0 min-h-screen">
          <Header />
          <main className="flex-1 px-4 sm:px-6 pb-8 max-w-7xl w-full mx-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
