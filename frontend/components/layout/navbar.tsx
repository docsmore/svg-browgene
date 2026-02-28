"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Globe, ListTodo, Compass, History, Settings } from "lucide-react";
import { cn } from "@/src/utils";

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
}

const navItems: NavItem[] = [
  { href: "/", label: "Dashboard", icon: Globe },
  { href: "/task-manager", label: "Tasks", icon: ListTodo },
  { href: "/explore", label: "Explore", icon: Compass },
  { href: "/executions", label: "Executions", icon: History },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="p-1.5 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
              <Globe className="h-5 w-5 text-browgene-blue" />
            </div>
            <span className="font-bold font-mono text-browgene-blue text-lg">
              BrowGene
            </span>
            <span className="text-xs font-mono text-browgene-dim bg-secondary px-1.5 py-0.5 rounded">
              v2
            </span>
          </Link>

          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary/15 text-browgene-blue"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 text-xs font-mono text-browgene-dim">
              <span className="flex items-center gap-1">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                API
              </span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
