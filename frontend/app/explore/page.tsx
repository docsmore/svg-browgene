"use client";

import { useState, useRef, useEffect, useCallback, Suspense } from "react";
import { motion } from "framer-motion";
import { EmbeddedNavbar } from "@/components/layout/embedded-wrapper";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Compass, Globe, Loader2, CheckCircle2, XCircle,
  Play, Sparkles, ArrowRight, Eye, GraduationCap
} from "lucide-react";
import type { ExplorationResult } from "@/src/types";

const POLL_INTERVAL_MS = 3000;

export default function ExplorePage() {
  const [url, setUrl] = useState("");
  const [taskDescription, setTaskDescription] = useState("");
  const [exploring, setExploring] = useState(false);
  const [result, setResult] = useState<ExplorationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [convertingToTask, setConvertingToTask] = useState(false);
  const [showTaskNameInput, setShowTaskNameInput] = useState(false);
  const [taskName, setTaskName] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const pollExploration = useCallback(
    (explorationId: string) => {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`/api/browgene/explorations/${explorationId}`);
          if (!res.ok) return; // keep polling, might not be ready yet
          const data: ExplorationResult = await res.json();

          if (data.status === "completed" || data.status === "failed") {
            stopPolling();
            setResult(data);
            setExploring(false);
            if (data.status === "failed" && data.error) {
              setError(data.error);
            }
          }
        } catch {
          // Network error during poll — keep trying
        }
      }, POLL_INTERVAL_MS);
    },
    [stopPolling]
  );

  const startExploration = async () => {
    if (!url || !taskDescription) return;
    setExploring(true);
    setError(null);
    setResult(null);
    stopPolling();

    try {
      const response = await fetch("/api/browgene/explore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_url: url,
          task: taskDescription,
          max_steps: 20,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        let detail = "Exploration failed";
        try {
          const err = JSON.parse(text);
          detail = err.detail || detail;
        } catch {
          detail = text || `HTTP ${response.status}`;
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const explorationId = data.exploration_id as string;

      // Start polling for results
      pollExploration(explorationId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setExploring(false);
    }
  };

  const handleConvertClick = () => {
    if (!result) return;
    const defaultName = taskDescription.slice(0, 50).replace(/[^a-zA-Z0-9 ]/g, "").trim().replace(/\s+/g, "-").toLowerCase() || "explored-task";
    setTaskName(defaultName);
    setShowTaskNameInput(true);
  };

  const convertToTask = async () => {
    if (!result || !taskName.trim()) return;
    setShowTaskNameInput(false);
    setConvertingToTask(true);
    try {
      const response = await fetch("/api/browgene/learn/from-exploration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exploration_id: result.exploration_id,
          task_name: taskName.trim(),
          description: taskDescription,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        let detail = "Conversion failed";
        try {
          const err = JSON.parse(text);
          detail = err.detail || detail;
        } catch {
          detail = text || `HTTP ${response.status}`;
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setError(null);
      setResult((prev) =>
        prev ? { ...prev, agent_output: `Task "${data.task_name}" created with ${data.steps_count} steps!` } : prev
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Conversion failed");
    } finally {
      setConvertingToTask(false);
    }
  };

  return (
    <div className="min-h-screen bg-browgene-gradient">
      <Suspense fallback={null}>
        <EmbeddedNavbar />
      </Suspense>

      <main className="container mx-auto px-4 py-8">
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-3xl font-bold font-mono mb-2 flex items-center gap-3 text-browgene-purple">
            Explore Mode
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            >
              <Compass className="h-6 w-6" />
            </motion.div>
          </h1>
          <p className="text-browgene-dim">
            Let AI discover and navigate unknown web workflows automatically
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.4 }}
          >
            <Card className="glass">
              <CardHeader>
                <CardTitle className="text-lg font-mono text-browgene-blue flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-browgene-yellow" />
                  AI Exploration
                </CardTitle>
                <CardDescription>
                  Provide a URL and describe the task. The AI agent will navigate
                  and discover the workflow.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-browgene-dim mb-1.5">
                    Target URL
                  </label>
                  <div className="relative">
                    <Globe className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="https://legacy-app.example.com"
                      className="pl-10"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-browgene-dim mb-1.5">
                    Task Description
                  </label>
                  <Textarea
                    value={taskDescription}
                    onChange={(e) => setTaskDescription(e.target.value)}
                    placeholder="Describe what you want the AI to do... e.g., 'Log in with admin credentials, navigate to policy search, find policy #12345, and extract the premium amount'"
                    rows={4}
                  />
                </div>

                <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                  <Button
                    className="w-full btn-gradient"
                    onClick={startExploration}
                    disabled={exploring || !url || !taskDescription}
                  >
                    {exploring ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Exploring...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Start Exploration
                      </>
                    )}
                  </Button>
                </motion.div>

                {error && (
                  <motion.div
                    className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive flex items-center gap-2"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <XCircle className="h-4 w-4 shrink-0" />
                    {error}
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Results Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            <Card className="glass h-full">
              <CardHeader>
                <CardTitle className="text-lg font-mono text-browgene-green flex items-center gap-2">
                  <Eye className="h-5 w-5" />
                  Exploration Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                {exploring ? (
                  <div className="flex flex-col items-center justify-center py-12 space-y-4">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <Compass className="h-12 w-12 text-browgene-purple" />
                    </motion.div>
                    <p className="text-sm text-browgene-dim font-mono">
                      AI agent exploring the web app...
                    </p>
                    <p className="text-xs text-browgene-dim">
                      This may take a few minutes
                    </p>
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      {result.success ? (
                        <Badge variant="success">
                          <CheckCircle2 className="h-3 w-3 mr-1" />
                          Success
                        </Badge>
                      ) : (
                        <Badge variant="destructive">
                          <XCircle className="h-3 w-3 mr-1" />
                          Failed
                        </Badge>
                      )}
                      <span className="text-xs text-browgene-dim font-mono">
                        {result.action_count} actions recorded
                      </span>
                    </div>

                    {result.recorded_actions.length > 0 && (
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {result.recorded_actions.map((action, i) => (
                          <motion.div
                            key={i}
                            className="flex items-center gap-2 p-2 rounded-lg bg-secondary/30 text-sm"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.05 }}
                          >
                            <span className="text-xs font-mono text-browgene-dim w-6 text-right shrink-0">
                              {i + 1}.
                            </span>
                            <Badge variant="outline" className="text-xs shrink-0">
                              {action.action_type}
                            </Badge>
                            <span className="text-browgene-dim truncate">
                              {action.description}
                            </span>
                          </motion.div>
                        ))}
                      </div>
                    )}

                    {result.success && !showTaskNameInput && (
                      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                        <Button
                          className="w-full"
                          variant="outline"
                          onClick={handleConvertClick}
                          disabled={convertingToTask}
                        >
                          {convertingToTask ? (
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          ) : (
                            <GraduationCap className="h-4 w-4 mr-2" />
                          )}
                          Convert to Reusable Task
                          <ArrowRight className="h-4 w-4 ml-2" />
                        </Button>
                      </motion.div>
                    )}

                    {showTaskNameInput && (
                      <motion.div
                        className="space-y-2"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <label className="block text-xs font-medium text-browgene-dim">
                          Task Name
                        </label>
                        <Input
                          value={taskName}
                          onChange={(e) => setTaskName(e.target.value)}
                          placeholder="my-reusable-task"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === "Enter") convertToTask();
                            if (e.key === "Escape") setShowTaskNameInput(false);
                          }}
                        />
                        <div className="flex gap-2">
                          <Button
                            className="flex-1 btn-gradient"
                            size="sm"
                            onClick={convertToTask}
                            disabled={!taskName.trim()}
                          >
                            <GraduationCap className="h-4 w-4 mr-1" />
                            Save Task
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setShowTaskNameInput(false)}
                          >
                            Cancel
                          </Button>
                        </div>
                      </motion.div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <Compass className="h-12 w-12 text-muted-foreground/20 mb-4" />
                    <p className="text-sm text-browgene-dim">
                      Results will appear here after exploration
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
