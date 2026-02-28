"use client";

import { Suspense, useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { EmbeddedNavbar } from "@/components/layout/embedded-wrapper";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Globe, Compass, GraduationCap, Play, ListTodo,
  History, Zap, MonitorSmartphone, Sparkles, ArrowRight,
  CheckCircle2, XCircle, Loader2
} from "lucide-react";
import type { HealthStatus } from "@/src/types";

const modeCards = [
  {
    icon: Compass,
    title: "Explore",
    description: "AI discovers unknown workflows using browser-use agent",
    href: "/explore",
    color: "text-browgene-purple",
    bgColor: "rgba(167, 139, 250, 0.15)",
    borderColor: "border-violet-500/20",
  },
  {
    icon: GraduationCap,
    title: "Learn",
    description: "Convert AI exploration into deterministic task steps",
    href: "/task-manager",
    color: "text-browgene-yellow",
    bgColor: "rgba(251, 191, 36, 0.15)",
    borderColor: "border-amber-500/20",
  },
  {
    icon: Play,
    title: "Execute",
    description: "Fast deterministic replay of recorded browser tasks",
    href: "/executions",
    color: "text-browgene-green",
    bgColor: "rgba(74, 222, 128, 0.15)",
    borderColor: "border-emerald-500/20",
  },
];

interface DashboardStats {
  tasks: number;
  executions: number;
  explorations: number;
  browserActive: boolean;
}

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [stats, setStats] = useState<DashboardStats>({
    tasks: 0, executions: 0, explorations: 0, browserActive: false,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthRes, tasksRes, execRes, exploreRes] = await Promise.allSettled([
          fetch("/api/browgene/health"),
          fetch("/api/browgene/tasks"),
          fetch("/api/browgene/executions"),
          fetch("/api/browgene/explorations"),
        ]);

        if (healthRes.status === "fulfilled" && healthRes.value.ok) {
          const data = await healthRes.value.json();
          setHealth(data);
        }

        let taskCount = 0;
        let execCount = 0;
        let exploreCount = 0;
        let browserActive = false;

        if (tasksRes.status === "fulfilled" && tasksRes.value.ok) {
          const data = await tasksRes.value.json();
          taskCount = data.tasks?.length || 0;
        }
        if (execRes.status === "fulfilled" && execRes.value.ok) {
          const data = await execRes.value.json();
          execCount = data.executions?.length || 0;
        }
        if (exploreRes.status === "fulfilled" && exploreRes.value.ok) {
          const data = await exploreRes.value.json();
          exploreCount = data.explorations?.length || 0;
        }
        if (health?.browser_active) browserActive = true;

        setStats({ tasks: taskCount, executions: execCount, explorations: exploreCount, browserActive });
      } catch {
        // API not available
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-browgene-gradient overflow-hidden">
      <Suspense fallback={null}>
        <EmbeddedNavbar />
      </Suspense>

      <main className="container mx-auto px-4 py-8">
        {/* Hero */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            className="inline-flex items-center gap-3 mb-4"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          >
            <Globe className="h-8 w-8 text-browgene-blue" />
            <h1 className="text-4xl font-bold font-mono text-browgene-blue text-shadow-glow">
              BrowGene v2
            </h1>
            <motion.div
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            >
              <Sparkles className="h-5 w-5 text-browgene-yellow" />
            </motion.div>
          </motion.div>
          <motion.p
            className="text-lg text-browgene-dim max-w-2xl mx-auto font-mono"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
          >
            Three-Mode Browser Automation: Explore, Learn, Execute
          </motion.p>

          {/* API Status */}
          <motion.div
            className="mt-4 inline-flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-full bg-card/50 border border-border/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            {health ? (
              <>
                <CheckCircle2 className="h-3.5 w-3.5 text-browgene-green" />
                <span className="text-browgene-green">API Connected</span>
                <span className="text-browgene-dim">|</span>
                <span className="text-browgene-dim">v{health.version}</span>
              </>
            ) : loading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 text-browgene-dim animate-spin" />
                <span className="text-browgene-dim">Connecting...</span>
              </>
            ) : (
              <>
                <XCircle className="h-3.5 w-3.5 text-browgene-red" />
                <span className="text-browgene-red">API Offline</span>
              </>
            )}
          </motion.div>
        </motion.div>

        {/* Three Mode Cards */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          {modeCards.map((card, index) => (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              whileHover={{ y: -6, transition: { duration: 0.2 } }}
            >
              <Link href={card.href}>
                <Card className={`glass cursor-pointer h-full hover:shadow-lg transition-shadow ${card.borderColor}`}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-3">
                      <motion.div
                        className="p-3 rounded-lg"
                        style={{ backgroundColor: card.bgColor }}
                        whileHover={{ scale: 1.1, rotate: 5 }}
                        transition={{ type: "spring", stiffness: 400 }}
                      >
                        <card.icon className={`h-6 w-6 ${card.color}`} />
                      </motion.div>
                      <div>
                        <CardTitle className={`text-lg font-mono ${card.color}`}>
                          {card.title}
                        </CardTitle>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-browgene-dim">{card.description}</p>
                    <div className="mt-3 flex items-center gap-1 text-xs text-browgene-blue">
                      <span>Open</span>
                      <ArrowRight className="h-3 w-3" />
                    </div>
                  </CardContent>
                </Card>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        {/* Stats Row */}
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
        >
          {[
            { label: "Tasks", value: stats.tasks, icon: ListTodo, color: "text-browgene-blue" },
            { label: "Executions", value: stats.executions, icon: Zap, color: "text-browgene-green" },
            { label: "Explorations", value: stats.explorations, icon: Compass, color: "text-browgene-purple" },
            { label: "Browser", value: stats.browserActive ? "Active" : "Idle", icon: MonitorSmartphone, color: stats.browserActive ? "text-browgene-green" : "text-browgene-dim" },
          ].map((stat) => (
            <Card key={stat.label} className="glass">
              <CardContent className="p-4 flex items-center gap-3">
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
                <div>
                  <p className="text-xs text-browgene-dim">{stat.label}</p>
                  <p className={`text-lg font-bold font-mono ${stat.color}`}>{stat.value}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.5 }}
        >
          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-base font-mono text-browgene-blue flex items-center gap-2">
                <ListTodo className="h-5 w-5" />
                Recent Tasks
              </CardTitle>
              <CardDescription>Your saved browser automation tasks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {stats.tasks === 0 ? (
                  <p className="text-sm text-browgene-dim py-4 text-center">
                    No tasks yet. Create one or explore a workflow.
                  </p>
                ) : (
                  <p className="text-sm text-browgene-dim">
                    {stats.tasks} task{stats.tasks !== 1 ? "s" : ""} available
                  </p>
                )}
              </div>
              <div className="mt-4">
                <Link href="/task-manager">
                  <Button size="sm" variant="outline" className="w-full">
                    <ListTodo className="h-4 w-4 mr-2" />
                    Manage Tasks
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>

          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-base font-mono text-browgene-purple flex items-center gap-2">
                <History className="h-5 w-5" />
                Recent Executions
              </CardTitle>
              <CardDescription>Task execution history</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {stats.executions === 0 ? (
                  <p className="text-sm text-browgene-dim py-4 text-center">
                    No executions yet. Run a task to see results here.
                  </p>
                ) : (
                  <p className="text-sm text-browgene-dim">
                    {stats.executions} execution{stats.executions !== 1 ? "s" : ""} recorded
                  </p>
                )}
              </div>
              <div className="mt-4">
                <Link href="/executions">
                  <Button size="sm" variant="outline" className="w-full">
                    <History className="h-4 w-4 mr-2" />
                    View Executions
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  );
}
