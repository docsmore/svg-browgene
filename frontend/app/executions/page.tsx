"use client";

import { useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { EmbeddedNavbar } from "@/components/layout/embedded-wrapper";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  History, Loader2, CheckCircle2, XCircle, Clock,
  ChevronRight, Zap, BarChart3, RefreshCw, AlertTriangle
} from "lucide-react";
import type { TaskExecution } from "@/src/types";

interface ExecutionListItem {
  execution_id: string;
  task_name: string;
  mode: string;
  status: string;
  steps_completed: number;
  steps_total: number;
  start_time?: string;
  end_time?: string;
  error?: string;
}

export default function ExecutionsPage() {
  const [executions, setExecutions] = useState<ExecutionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    fetchExecutions();
  }, []);

  const fetchExecutions = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/browgene/executions");
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const data = await response.json();
      if (data.executions) {
        setExecutions(data.executions);
      }
      setApiError(null);
    } catch (error) {
      console.error("Failed to fetch executions:", error);
      setApiError(
        "Cannot connect to BrowGene API at localhost:8200. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="success">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        );
      case "running":
        return (
          <Badge variant="info">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Running
          </Badge>
        );
      case "partial":
        return (
          <Badge variant="warning">
            <AlertTriangle className="h-3 w-3 mr-1" />
            Partial
          </Badge>
        );
      default:
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            {status}
          </Badge>
        );
    }
  };

  const getModeBadge = (mode: string) => {
    switch (mode) {
      case "explore":
        return <Badge variant="info">Explore</Badge>;
      case "learn":
        return <Badge variant="warning">Learn</Badge>;
      case "execute":
        return <Badge variant="success">Execute</Badge>;
      default:
        return <Badge variant="secondary">{mode}</Badge>;
    }
  };

  return (
    <div className="min-h-screen bg-browgene-gradient">
      <Suspense fallback={null}>
        <EmbeddedNavbar />
      </Suspense>

      <main className="container mx-auto px-4 py-8">
        <motion.div
          className="flex items-center justify-between mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div>
            <h1 className="text-3xl font-bold font-mono mb-2 flex items-center gap-3 text-browgene-blue">
              Execution History
              <motion.div
                animate={{ rotate: [0, -10, 10, 0] }}
                transition={{ duration: 3, repeat: Infinity, repeatDelay: 2 }}
              >
                <BarChart3 className="h-6 w-6 text-browgene-cyan" />
              </motion.div>
            </h1>
            <p className="text-browgene-dim">
              View results and details of all task executions
            </p>
          </div>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button variant="outline" onClick={fetchExecutions}>
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </motion.div>
        </motion.div>

        {apiError && (
          <motion.div
            className="mb-6 p-4 rounded-lg bg-destructive/10 border border-destructive/30 flex items-center gap-3"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <XCircle className="h-5 w-5 text-destructive shrink-0" />
            <div>
              <p className="text-sm font-medium text-destructive">Backend Offline</p>
              <p className="text-xs text-destructive/70">{apiError}</p>
            </div>
          </motion.div>
        )}

        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div
              key="loading"
              className="flex items-center justify-center py-16"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
            </motion.div>
          ) : executions.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <Card className="glass">
                <CardContent className="py-16">
                  <div className="text-center text-muted-foreground">
                    <History className="h-16 w-16 mx-auto mb-4 opacity-30" />
                    <p className="text-lg font-medium">No executions yet</p>
                    <p className="text-sm mb-4">
                      Execute a task to see results here
                    </p>
                    <Link href="/task-manager">
                      <Button>
                        <Zap className="h-4 w-4 mr-2" />
                        Go to Task Manager
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              key="executions"
              className="space-y-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {executions.map((exec, index) => (
                <motion.div
                  key={exec.execution_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05, duration: 0.3 }}
                  whileHover={{ x: 4, transition: { duration: 0.15 } }}
                >
                  <Link href={`/executions/${exec.execution_id}`}>
                    <Card className="glass hover:shadow-lg transition-shadow cursor-pointer">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4 min-w-0">
                            <div>
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-mono text-browgene-blue font-medium truncate">
                                  {exec.task_name}
                                </span>
                                {getModeBadge(exec.mode)}
                                {getStatusBadge(exec.status)}
                              </div>
                              <div className="flex items-center gap-3 text-xs text-browgene-dim">
                                <span className="font-mono">
                                  {exec.execution_id.substring(0, 8)}...
                                </span>
                                {exec.start_time && (
                                  <span className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {new Date(exec.start_time).toLocaleString()}
                                  </span>
                                )}
                                <span>
                                  {exec.steps_completed}/{exec.steps_total} steps
                                </span>
                              </div>
                              {exec.error && (
                                <p className="text-xs text-destructive mt-1 truncate max-w-md">
                                  {exec.error}
                                </p>
                              )}
                            </div>
                          </div>
                          <ChevronRight className="h-5 w-5 text-muted-foreground shrink-0" />
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
