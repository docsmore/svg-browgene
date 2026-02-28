"use client";

import { useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { EmbeddedNavbar } from "@/components/layout/embedded-wrapper";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import {
  ListTodo, Play, Edit, Trash2, Loader2, Plus, Clock,
  MousePointer, Keyboard, Search, Globe, Zap, Eye,
  ArrowRight, Compass, Type, ScrollText, XCircle,
  CheckCircle2, AlertTriangle, Brain, Bot
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { TaskListItem } from "@/src/types";

interface ExecutionResult {
  type: "success" | "error";
  title: string;
  message: string;
  executionId?: string;
}

export default function TaskManagerPage() {
  const router = useRouter();
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [executing, setExecuting] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [showCreateAgentic, setShowCreateAgentic] = useState(false);
  const [newAgenticName, setNewAgenticName] = useState("");
  const [newAgenticGoal, setNewAgenticGoal] = useState("");
  const [newAgenticUrl, setNewAgenticUrl] = useState("");
  const [creatingAgentic, setCreatingAgentic] = useState(false);

  useEffect(() => {
    fetchTasks();
  }, []);

  const fetchTasks = async () => {
    try {
      const response = await fetch("/api/browgene/tasks");
      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }
      const data = await response.json();
      if (data.tasks) {
        setTasks(data.tasks);
      }
      setApiError(null);
    } catch (error) {
      console.error("Failed to fetch tasks:", error);
      setApiError(
        "Cannot connect to BrowGene API at localhost:8200. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const deleteTask = async (name: string) => {
    try {
      const response = await fetch(`/api/browgene/tasks/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      if (response.ok) {
        setTasks(tasks.filter((t) => t.name !== name));
      }
    } catch (error) {
      console.error("Failed to delete task:", error);
    } finally {
      setDeleteTarget(null);
    }
  };

  const executeTask = async (name: string) => {
    setExecuting(name);
    try {
      const response = await fetch("/api/browgene/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_name: name, parameters: {} }),
      });
      if (response.ok) {
        const data = await response.json();

        // Agentic tasks run async — poll for results
        if (data.status === "running" && data.execution_id) {
          await pollExecution(data.execution_id, name);
          return;
        }

        setExecutionResult({
          type: data.status === "failed" ? "error" : "success",
          title: data.status === "failed" ? "Execution Failed" : "Execution Complete",
          message: data.status === "failed"
            ? (data.error || "Task failed")
            : `Task "${name}" executed successfully.\nStatus: ${data.status}`,
          executionId: data.execution_id,
        });
      } else {
        const err = await response.json();
        setExecutionResult({
          type: "error",
          title: "Execution Failed",
          message: err.detail || "Unknown error occurred",
        });
      }
    } catch (error) {
      console.error("Execute failed:", error);
      setExecutionResult({
        type: "error",
        title: "Connection Error",
        message: "Failed to connect to the backend. Make sure the server is running.",
      });
    } finally {
      setExecuting(null);
    }
  };

  const pollExecution = async (executionId: string, taskName: string) => {
    const poll = async (): Promise<void> => {
      try {
        const res = await fetch(`/api/browgene/executions/${executionId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.status === "running") {
          await new Promise((r) => setTimeout(r, 3000));
          return poll();
        }

        const agentOutput = data.extracted_data?.[0]?.agent_output;
        setExecutionResult({
          type: data.status === "completed" ? "success" : "error",
          title: data.status === "completed" ? "Agentic Task Complete" : "Agentic Task Failed",
          message: data.status === "completed"
            ? (agentOutput || `Task "${taskName}" completed with ${data.steps_completed} agent steps.`)
            : (data.error || "Task failed"),
          executionId: data.execution_id,
        });
      } catch {
        setExecutionResult({
          type: "error",
          title: "Polling Error",
          message: "Lost connection while waiting for agentic task results.",
        });
      } finally {
        setExecuting(null);
      }
    };
    await poll();
  };

  const createAgenticTask = async () => {
    if (!newAgenticName.trim() || !newAgenticGoal.trim()) return;
    setCreatingAgentic(true);
    try {
      const response = await fetch("/api/browgene/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newAgenticName.trim(),
          description: newAgenticGoal.trim(),
          mode: "agentic",
          goal: newAgenticGoal.trim(),
          start_url: newAgenticUrl.trim() || undefined,
          steps: [],
          tags: ["agentic"],
        }),
      });
      if (response.ok) {
        setShowCreateAgentic(false);
        setNewAgenticName("");
        setNewAgenticGoal("");
        setNewAgenticUrl("");
        router.push(`/task-manager/${encodeURIComponent(newAgenticName.trim())}`);
      }
    } catch (error) {
      console.error("Failed to create agentic task:", error);
    } finally {
      setCreatingAgentic(false);
    }
  };

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case "click":
      case "right_click":
      case "double_click":
        return <MousePointer className="h-3 w-3" />;
      case "fill":
      case "type":
      case "keyboard":
        return <Keyboard className="h-3 w-3" />;
      case "navigate":
        return <Globe className="h-3 w-3" />;
      case "scroll":
        return <ScrollText className="h-3 w-3" />;
      case "extract":
        return <Eye className="h-3 w-3" />;
      case "wait":
      case "wait_for":
        return <Clock className="h-3 w-3" />;
      default:
        return <Type className="h-3 w-3" />;
    }
  };

  const filteredTasks = tasks.filter(
    (task) =>
      task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      task.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

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
              Task Manager
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
              >
                <Zap className="h-6 w-6 text-browgene-yellow" />
              </motion.div>
            </h1>
            <p className="text-browgene-dim">
              Manage and execute your browser automation tasks
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/explore">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button variant="outline">
                  <Compass className="h-4 w-4 mr-2" />
                  Explore New
                </Button>
              </motion.div>
            </Link>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button variant="outline" className="border-purple-500/30 hover:border-purple-500/60" onClick={() => setShowCreateAgentic(true)}>
                <Brain className="h-4 w-4 mr-2 text-purple-400" />
                New Agentic Task
              </Button>
            </motion.div>
          </div>
        </motion.div>

        {/* API Error Banner */}
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

        {/* Search */}
        <motion.div
          className="mb-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.4 }}
        >
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search tasks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-ring transition-all"
            />
          </div>
        </motion.div>

        {/* Tasks Grid */}
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
          ) : filteredTasks.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <Card className="glass">
                <CardContent className="py-16">
                  <div className="text-center text-muted-foreground">
                    <ListTodo className="h-16 w-16 mx-auto mb-4 opacity-30" />
                    {searchQuery ? (
                      <>
                        <p className="text-lg font-medium">No tasks found</p>
                        <p className="text-sm">Try a different search term</p>
                      </>
                    ) : (
                      <>
                        <p className="text-lg font-medium">No tasks yet</p>
                        <p className="text-sm mb-4">
                          Explore a workflow or create a task manually
                        </p>
                        <Link href="/explore">
                          <Button>
                            <Compass className="h-4 w-4 mr-2" />
                            Start Exploring
                          </Button>
                        </Link>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              key="tasks"
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {filteredTasks.map((task, index) => (
                <motion.div
                  key={task.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.08, duration: 0.4 }}
                  whileHover={{ y: -4, transition: { duration: 0.2 } }}
                >
                  <Card className="glass hover:shadow-lg transition-shadow h-full flex flex-col">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <CardTitle className="text-lg truncate text-browgene-blue font-mono">
                            {task.name}
                          </CardTitle>
                          <CardDescription className="line-clamp-2 mt-1">
                            {task.description || "No description"}
                          </CardDescription>
                        </div>
                        <Badge
                          variant={task.source === "explored" ? "info" : task.source === "recorded" ? "warning" : "secondary"}
                          className="ml-2 shrink-0"
                        >
                          {task.source || "manual"}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="flex-1 flex flex-col">
                      <div className="space-y-4 flex-1 flex flex-col">
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          {task.mode === "agentic" ? (
                            <span className="flex items-center gap-1 text-purple-400">
                              <Brain className="h-3.5 w-3.5" />
                              Agentic
                            </span>
                          ) : (
                            <span className="flex items-center gap-1">
                              <ListTodo className="h-4 w-4" />
                              {task.steps} steps
                            </span>
                          )}
                          {task.start_url && (
                            <span className="flex items-center gap-1 truncate max-w-[140px]" title={task.start_url}>
                              <Globe className="h-4 w-4 shrink-0" />
                              {new URL(task.start_url).hostname}
                            </span>
                          )}
                        </div>

                        {task.created_at && (
                          <div className="text-xs text-browgene-dim flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(task.created_at).toLocaleDateString()}
                          </div>
                        )}

                        {/* Actions */}
                        <div className="flex items-center gap-2 pt-2 border-t border-border/30 mt-auto">
                          <motion.div className="flex-1" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                            <Button
                              size="sm"
                              className="w-full btn-gradient"
                              onClick={() => executeTask(task.name)}
                              disabled={executing === task.name}
                            >
                              {executing === task.name ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                              ) : (
                                <Play className="h-4 w-4 mr-1" />
                              )}
                              Execute
                            </Button>
                          </motion.div>
                          <Link href={`/task-manager/${encodeURIComponent(task.name)}`}>
                            <Button size="sm" variant="outline">
                              <Edit className="h-4 w-4" />
                            </Button>
                          </Link>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setDeleteTarget(task.name)}
                          >
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Delete Confirmation Dialog */}
        <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                Delete Task
              </AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete <span className="font-semibold text-foreground">&quot;{deleteTarget}&quot;</span>? This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                onClick={() => deleteTarget && deleteTask(deleteTarget)}
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Execution Result Dialog */}
        <Dialog open={!!executionResult} onOpenChange={(open) => !open && setExecutionResult(null)}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                {executionResult?.type === "success" ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : (
                  <XCircle className="h-5 w-5 text-destructive" />
                )}
                {executionResult?.title}
              </DialogTitle>
              <DialogDescription className="whitespace-pre-line">
                {executionResult?.message}
              </DialogDescription>
            </DialogHeader>
            {executionResult?.executionId && (
              <div className="rounded-lg bg-secondary/30 p-3 font-mono text-xs">
                <span className="text-muted-foreground">Execution ID: </span>
                <span className="text-browgene-cyan">{executionResult.executionId}</span>
              </div>
            )}
            <DialogFooter>
              {executionResult?.executionId && (
                <Button
                  variant="outline"
                  onClick={() => {
                    router.push(`/executions/${executionResult.executionId}`);
                    setExecutionResult(null);
                  }}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  View Details
                </Button>
              )}
              <Button onClick={() => setExecutionResult(null)}>
                OK
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
        {/* Create Agentic Task Dialog */}
        <Dialog open={showCreateAgentic} onOpenChange={setShowCreateAgentic}>
          <DialogContent className="sm:max-w-lg">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-400" />
                Create Agentic Task
              </DialogTitle>
              <DialogDescription>
                Create an AI-driven research task. The agent will freely browse the web to accomplish the goal.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-1 block">Task Name</label>
                <Input
                  value={newAgenticName}
                  onChange={(e) => setNewAgenticName(e.target.value)}
                  placeholder="e.g. find-funeral-info-heather-bagg"
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Goal / Prompt</label>
                <Textarea
                  value={newAgenticGoal}
                  onChange={(e) => setNewAgenticGoal(e.target.value)}
                  placeholder='e.g. "Go find funeral information of Heather Bagg and bring back results for downstream processing"'
                  className="min-h-[100px]"
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-1 block">Start URL (optional)</label>
                <Input
                  value={newAgenticUrl}
                  onChange={(e) => setNewAgenticUrl(e.target.value)}
                  placeholder="https://google.com"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateAgentic(false)}>Cancel</Button>
              <Button
                onClick={createAgenticTask}
                disabled={creatingAgentic || !newAgenticName.trim() || !newAgenticGoal.trim()}
                className="bg-purple-600 hover:bg-purple-700"
              >
                {creatingAgentic ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Brain className="h-4 w-4 mr-2" />
                )}
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </main>
    </div>
  );
}
