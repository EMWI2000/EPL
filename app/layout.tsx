import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "FPL HoldPlanner",
  description:
    "Databaseret holdoptimering til Fantasy Premier League med gennemsigtige projektioner.",
  applicationName: "FPL HoldPlanner",
  robots: {
    index: false,
    follow: false,
  },
};

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#153d35",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="da">
      <body>{children}</body>
    </html>
  );
}
