import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BrowGene v2 - Browser Automation",
  description: "Three-Mode Browser Automation: Explore, Learn, Execute",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background antialiased">
        {children}
      </body>
    </html>
  );
}
