"use client";

import { useSearchParams } from "next/navigation";
import { Navbar } from "./navbar";

export function EmbeddedNavbar() {
  const searchParams = useSearchParams();
  const isEmbedded = searchParams.get("embedded") === "true";

  if (isEmbedded) {
    return null;
  }

  return <Navbar />;
}
