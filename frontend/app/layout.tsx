import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Chat",
  description: "Minimal Next.js frontend for streaming RAG chat",
  icons: {
    icon: "/rag-mark.svg",
    shortcut: "/rag-mark.svg",
    apple: "/rag-mark.svg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
