import "./globals.css";
import type { Metadata } from "next";
import { LanguageProvider } from "@/components/LanguageContext";

export const metadata: Metadata = {
  title: "BIM Coordinator",
  description: "Voice-driven AI architect — Phase 1 walking-skeleton dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <LanguageProvider>{children}</LanguageProvider>
      </body>
    </html>
  );
}
