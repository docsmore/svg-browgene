"use client";

import { useState, useEffect, Suspense, use, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { EmbeddedNavbar } from "@/components/layout/embedded-wrapper";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  ArrowLeft, Play, Loader2, Globe, Clock, Tag, ListTodo,
  MousePointer, Keyboard, Eye, ScrollText, Type, ChevronDown,
  ChevronUp, Zap, Settings, CheckCircle2, XCircle, Save,
  Trash2, Plus, GripVertical, Code, AlertTriangle,
  Navigation, Filter, Download, Brain, Search, Bot
} from "lucide-react";
import type { BrowserTask, BrowserStep, TaskMode } from "@/src/types";

/* ── Sortable wrapper ─────────────────────────────────────────── */
interface SortableStepItemProps {
  id: string;
  children: (dragHandleProps: React.HTMLAttributes<HTMLElement>) => React.ReactNode;
}

function SortableStepItem({ id, children }: SortableStepItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 1000 : "auto",
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes}>
      {children(listeners || {})}
    </div>
  );
}

/* ── Types ────────────────────────────────────────────────────── */
interface ExecutionResult {
  type: "success" | "error";
  title: string;
  message: string;
  executionId?: string;
}

interface TaskDetailPageProps {
  params: Promise<{ taskName: string }>;
}

const STEP_TYPES = [
  "navigate", "click", "right_click", "double_click", "fill", "type",
  "select", "scroll", "wait", "wait_for", "extract", "keyboard",
  "screenshot", "hover", "done",
] as const;

/* ── Main component ───────────────────────────────────────────── */
export default function TaskDetailPage({ params }: TaskDetailPageProps) {
  const { taskName } = use(params);
  const decodedName = decodeURIComponent(taskName);
  const router = useRouter();

  const [task, setTask] = useState<BrowserTask | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const [jsonEditMode, setJsonEditMode] = useState<Set<number>>(new Set());
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [deleteStepIndex, setDeleteStepIndex] = useState<number | null>(null);

  // DnD sensors
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  useEffect(() => {
    fetchTask();
  }, [decodedName]);

  const fetchTask = async () => {
    try {
      const response = await fetch(`/api/browgene/tasks/${encodeURIComponent(decodedName)}`);
      if (!response.ok) throw new Error("Failed to fetch");
      const data = await response.json();
      setTask(data);
    } catch (error) {
      console.error("Failed to fetch task:", error);
    } finally {
      setLoading(false);
    }
  };

  /* ── Save task ───────────────────────────────────────────────── */
  const saveTask = async () => {
    if (!task) return;
    setSaving(true);
    try {
      const response = await fetch(`/api/browgene/tasks/${encodeURIComponent(decodedName)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steps: task.steps.map((s) => ({
            step_type: s.step_type,
            params: s.params,
            description: s.description,
            delay_ms: s.delay_ms,
            take_screenshot: s.take_screenshot,
            on_failure: s.on_failure,
            max_retries: s.max_retries,
            timeout_ms: s.timeout_ms,
          })),
          description: task.description,
          start_url: task.start_url,
          tags: task.tags,
          mode: task.mode,
          goal: task.goal,
          max_agent_steps: task.max_agent_steps,
        }),
      });
      if (response.ok) {
        setHasChanges(false);
      }
    } catch (error) {
      console.error("Failed to save task:", error);
    } finally {
      setSaving(false);
    }
  };

  /* ── Execute task ────────────────────────────────────────────── */
  const executeTask = async () => {
    if (!task) return;
    setExecuting(true);
    try {
      const response = await fetch("/api/browgene/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_name: task.name, parameters: {} }),
      });
      if (response.ok) {
        const data = await response.json();

        // Agentic tasks run async — poll for results
        if (data.status === "running" && data.execution_id) {
          await pollExecution(data.execution_id);
          return;
        }

        setExecutionResult({
          type: data.status === "failed" ? "error" : "success",
          title: data.status === "failed" ? "Execution Failed" : "Execution Complete",
          message: data.status === "failed"
            ? (data.error || "Task failed")
            : `Task "${task.name}" executed successfully.\nStatus: ${data.status}`,
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
        message: "Failed to connect to the backend.",
      });
    } finally {
      setExecuting(false);
    }
  };

  const pollExecution = async (executionId: string) => {
    const poll = async (): Promise<void> => {
      try {
        const res = await fetch(`/api/browgene/executions/${executionId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.status === "running") {
          await new Promise((r) => setTimeout(r, 3000));
          return poll();
        }

        // Execution finished — show result
        const agentOutput = data.extracted_data?.[0]?.agent_output;
        setExecutionResult({
          type: data.status === "completed" ? "success" : "error",
          title: data.status === "completed" ? "Agentic Task Complete" : "Agentic Task Failed",
          message: data.status === "completed"
            ? (agentOutput || `Task completed with ${data.steps_completed} agent steps.`)
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
        setExecuting(false);
      }
    };
    await poll();
  };

  /* ── Step mutations ──────────────────────────────────────────── */
  const updateStep = (index: number, updates: Partial<BrowserStep>) => {
    if (!task) return;
    const newSteps = [...task.steps];
    newSteps[index] = { ...newSteps[index], ...updates };
    setTask({ ...task, steps: newSteps });
    setHasChanges(true);
  };

  const updateStepParams = (index: number, paramUpdates: Record<string, unknown>) => {
    if (!task) return;
    const newSteps = [...task.steps];
    newSteps[index] = {
      ...newSteps[index],
      params: { ...newSteps[index].params, ...paramUpdates },
    };
    setTask({ ...task, steps: newSteps });
    setHasChanges(true);
  };

  const deleteStep = () => {
    if (!task || deleteStepIndex === null) return;
    const newSteps = task.steps.filter((_, i) => i !== deleteStepIndex);
    setTask({ ...task, steps: newSteps });
    setDeleteStepIndex(null);
    setHasChanges(true);
  };

  const addStep = (afterIndex: number, stepType: string) => {
    if (!task) return;

    const templates: Record<string, BrowserStep> = {
      wait: {
        step_type: "wait", params: { seconds: 2 }, description: "Wait 2 seconds",
        delay_ms: 200, take_screenshot: false, on_failure: "continue", max_retries: 1, timeout_ms: 30000,
      },
      click: {
        step_type: "click", params: { element_text: "", element_tag: "button" }, description: "Click element",
        delay_ms: 200, take_screenshot: false, on_failure: "ai_fallback", max_retries: 1, timeout_ms: 30000,
      },
      fill: {
        step_type: "fill", params: { value: "", element_tag: "input", element_attrs: {} }, description: "Fill input",
        delay_ms: 200, take_screenshot: false, on_failure: "ai_fallback", max_retries: 1, timeout_ms: 30000,
      },
      navigate: {
        step_type: "navigate", params: { url: "https://" }, description: "Navigate to URL",
        delay_ms: 200, take_screenshot: false, on_failure: "stop", max_retries: 1, timeout_ms: 30000,
      },
      select: {
        step_type: "select", params: { text: "", element_tag: "select" }, description: "Select dropdown option",
        delay_ms: 200, take_screenshot: false, on_failure: "ai_fallback", max_retries: 1, timeout_ms: 30000,
      },
      extract: {
        step_type: "extract", params: { query: "" }, description: "Extract data from page",
        delay_ms: 200, take_screenshot: false, on_failure: "continue", max_retries: 1, timeout_ms: 30000,
      },
      keyboard: {
        step_type: "keyboard", params: { key: "Enter" }, description: "Press key",
        delay_ms: 200, take_screenshot: false, on_failure: "continue", max_retries: 1, timeout_ms: 30000,
      },
    };

    const newStep = templates[stepType] || {
      step_type: stepType, params: {}, description: stepType,
      delay_ms: 200, take_screenshot: false, on_failure: "ai_fallback", max_retries: 1, timeout_ms: 30000,
    };

    const newSteps = [...task.steps];
    newSteps.splice(afterIndex + 1, 0, newStep);
    setTask({ ...task, steps: newSteps });
    setExpandedSteps((prev) => new Set([...prev, afterIndex + 1]));
    setHasChanges(true);
  };

  const updateStepFromJson = (index: number, jsonStr: string) => {
    try {
      const parsed = JSON.parse(jsonStr) as BrowserStep;
      if (!task) return;
      const newSteps = [...task.steps];
      newSteps[index] = parsed;
      setTask({ ...task, steps: newSteps });
      setHasChanges(true);
    } catch {
      // Invalid JSON — don't update
    }
  };

  /* ── Drag & Drop ─────────────────────────────────────────────── */
  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (!over || active.id === over.id || !task) return;
      const oldIndex = task.steps.findIndex((_, i) => `step-${i}` === active.id);
      const newIndex = task.steps.findIndex((_, i) => `step-${i}` === over.id);
      if (oldIndex === -1 || newIndex === -1) return;
      const newSteps = arrayMove(task.steps, oldIndex, newIndex);
      setTask({ ...task, steps: newSteps });
      setHasChanges(true);
    },
    [task]
  );

  /* ── Helpers ─────────────────────────────────────────────────── */
  const toggleStep = (index: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const toggleJsonMode = (index: number) => {
    setJsonEditMode((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case "click": case "right_click": case "double_click": case "hover":
        return <MousePointer className="h-4 w-4" />;
      case "fill": case "type": case "keyboard":
        return <Keyboard className="h-4 w-4" />;
      case "navigate":
        return <Globe className="h-4 w-4" />;
      case "scroll":
        return <ScrollText className="h-4 w-4" />;
      case "extract":
        return <Eye className="h-4 w-4" />;
      case "wait": case "wait_for":
        return <Clock className="h-4 w-4" />;
      case "select":
        return <Filter className="h-4 w-4" />;
      case "done":
        return <CheckCircle2 className="h-4 w-4" />;
      default:
        return <Type className="h-4 w-4" />;
    }
  };

  const getStepSummary = (step: BrowserStep): string => {
    const p = step.params as Record<string, unknown>;
    if (p.element_text) return `"${String(p.element_text)}"`;
    if (p.url) return String(p.url);
    if (p.value) return `"${String(p.value).slice(0, 30)}"`;
    if (p.text) return `"${String(p.text).slice(0, 30)}"`;
    if (p.seconds) return `${p.seconds}s`;
    if (p.key) return String(p.key);
    if (p.query) return `"${String(p.query).slice(0, 30)}"`;
    return "";
  };

  /* ── Loading / Not found ─────────────────────────────────────── */
  if (loading) {
    return (
      <div className="min-h-screen bg-browgene-gradient flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!task) {
    return (
      <div className="min-h-screen bg-browgene-gradient">
        <Suspense fallback={null}>
          <EmbeddedNavbar />
        </Suspense>
        <main className="container mx-auto px-4 py-8 text-center">
          <p className="text-muted-foreground text-lg">Task not found</p>
          <Link href="/task-manager">
            <Button variant="outline" className="mt-4">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Tasks
            </Button>
          </Link>
        </main>
      </div>
    );
  }

  /* ── Render ──────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen bg-browgene-gradient">
      <Suspense fallback={null}>
        <EmbeddedNavbar />
      </Suspense>

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Link href="/task-manager" className="inline-flex items-center gap-1 text-sm text-browgene-dim hover:text-browgene-blue transition-colors mb-4">
            <ArrowLeft className="h-4 w-4" />
            Back to Tasks
          </Link>

          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold font-mono text-browgene-blue mb-2">
                {task.name}
              </h1>
              <p className="text-browgene-dim max-w-2xl">{task.description || "No description"}</p>
              <div className="flex items-center gap-3 mt-3">
                <Badge variant={task.mode === "agentic" ? "default" : "secondary"} className={task.mode === "agentic" ? "bg-purple-600 hover:bg-purple-700" : ""}>
                  {task.mode === "agentic" ? (
                    <><Brain className="h-3 w-3 mr-1" /> Agentic</>
                  ) : (
                    <><ListTodo className="h-3 w-3 mr-1" /> {task.steps.length} steps</>
                  )}
                </Badge>
                <Badge variant="info">{task.source || "manual"}</Badge>
                {task.start_url && (
                  <Badge variant="outline" className="font-mono text-xs">
                    <Globe className="h-3 w-3 mr-1" />
                    {task.start_url}
                  </Badge>
                )}
                {task.tags?.map((tag) => (
                  <Badge key={tag} variant="outline">
                    <Tag className="h-3 w-3 mr-1" />
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={saveTask}
                disabled={saving || !hasChanges}
              >
                {saving ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-2" />
                )}
                {hasChanges ? "Save Changes" : "Saved"}
              </Button>
              <Button className="btn-gradient" onClick={executeTask} disabled={executing}>
                {executing ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Execute
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Agentic Goal Editor */}
        {task.mode === "agentic" && (
          <motion.div
            className="mb-6"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="glass border-purple-500/30">
              <CardHeader>
                <CardTitle className="text-base font-mono text-purple-400 flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Agentic Task — AI Goal
                </CardTitle>
                <CardDescription>
                  The AI agent will freely browse the web to accomplish this goal and return results
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-xs font-medium text-browgene-dim mb-1 block">Goal / Prompt</label>
                  <Textarea
                    value={task.goal || ""}
                    onChange={(e) => {
                      setTask({ ...task, goal: e.target.value });
                      setHasChanges(true);
                    }}
                    placeholder='e.g. "Go find funeral information of Heather Bagg and bring back results for downstream processing"'
                    className="min-h-[120px] bg-card/50 border-purple-500/20 focus:border-purple-500/50"
                  />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-xs font-medium text-browgene-dim mb-1 block">Start URL (optional)</label>
                    <Input
                      value={task.start_url || ""}
                      onChange={(e) => {
                        setTask({ ...task, start_url: e.target.value });
                        setHasChanges(true);
                      }}
                      placeholder="https://google.com"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-browgene-dim mb-1 block">Max Agent Steps</label>
                    <Input
                      type="number"
                      value={task.max_agent_steps || 25}
                      onChange={(e) => {
                        setTask({ ...task, max_agent_steps: parseInt(e.target.value) || 25 });
                        setHasChanges(true);
                      }}
                      min={5}
                      max={100}
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-browgene-dim mb-1 block">Switch Mode</label>
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => {
                        setTask({ ...task, mode: "deterministic" });
                        setHasChanges(true);
                      }}
                    >
                      <Zap className="h-4 w-4 mr-2" />
                      Switch to Deterministic
                    </Button>
                  </div>
                </div>
                <div className="rounded-lg bg-purple-500/5 border border-purple-500/20 p-3 text-xs text-browgene-dim">
                  <div className="flex items-start gap-2">
                    <Bot className="h-4 w-4 mt-0.5 text-purple-400 shrink-0" />
                    <div>
                      <p className="font-medium text-purple-400 mb-1">How it works</p>
                      <p>When executed, the AI agent (GPT-4o + browser-use) will autonomously browse the web to accomplish your goal.
                        It will navigate pages, click links, fill forms, and extract information — then return the results for downstream processing.
                        No predefined steps needed.</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Steps Card — Deterministic mode */}
        {task.mode !== "agentic" && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="glass">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-base font-mono text-browgene-blue flex items-center gap-2">
                    <Zap className="h-4 w-4" />
                    Workflow Steps ({task.steps.length})
                  </CardTitle>
                  <CardDescription>
                    Drag to reorder, expand to edit, or add new steps
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setTask({ ...task, mode: "agentic", goal: task.description });
                    setHasChanges(true);
                  }}
                >
                  <Brain className="h-4 w-4 mr-1" />
                  Switch to Agentic
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={handleDragEnd}
              >
                <SortableContext
                  items={task.steps.map((_, i) => `step-${i}`)}
                  strategy={verticalListSortingStrategy}
                >
                  <div className="space-y-2">
                    {task.steps.map((step, index) => {
                      const isExpanded = expandedSteps.has(index);
                      const isJsonMode = jsonEditMode.has(index);
                      const params = step.params as Record<string, unknown>;
                      const summary = getStepSummary(step);

                      return (
                        <SortableStepItem key={`step-${index}`} id={`step-${index}`}>
                          {(dragHandleProps) => (
                            <div className="rounded-lg border border-border/30 overflow-hidden bg-card/50">
                              {/* Step header */}
                              <div
                                className="flex items-center gap-3 p-3 hover:bg-secondary/20 transition-colors cursor-pointer"
                                onClick={() => toggleStep(index)}
                              >
                                <div
                                  {...dragHandleProps}
                                  className="touch-none"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <GripVertical className="h-5 w-5 text-muted-foreground cursor-grab hover:text-foreground transition-colors" />
                                </div>
                                <span className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center text-xs font-mono font-medium shrink-0">
                                  {index + 1}
                                </span>
                                <span className="text-browgene-cyan">
                                  {getStepIcon(step.step_type)}
                                </span>
                                <Badge variant="outline" className="text-xs font-mono shrink-0">
                                  {step.step_type}
                                </Badge>
                                <span className="text-sm text-browgene-dim flex-1 truncate">
                                  {step.description}
                                  {summary && (
                                    <span className="ml-1 text-browgene-cyan">{summary}</span>
                                  )}
                                </span>
                                <span className="text-xs text-browgene-dim font-mono shrink-0">
                                  {step.delay_ms}ms
                                </span>
                                <div className="flex items-center gap-1">
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleJsonMode(index);
                                    }}
                                  >
                                    {isJsonMode ? <Eye className="h-3.5 w-3.5" /> : <Code className="h-3.5 w-3.5" />}
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setDeleteStepIndex(index);
                                    }}
                                  >
                                    <Trash2 className="h-3.5 w-3.5 text-destructive" />
                                  </Button>
                                  {isExpanded ? (
                                    <ChevronUp className="h-4 w-4 text-muted-foreground" />
                                  ) : (
                                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                                  )}
                                </div>
                              </div>

                              {/* Expanded editor */}
                              {isExpanded && (
                                <div className="border-t border-border/20 bg-secondary/10 p-4">
                                  {isJsonMode ? (
                                    <Textarea
                                      value={JSON.stringify(step, null, 2)}
                                      onChange={(e) => updateStepFromJson(index, e.target.value)}
                                      className="font-mono text-xs min-h-[200px] bg-card"
                                    />
                                  ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                      {/* Step Type */}
                                      <div>
                                        <label className="text-xs font-medium text-browgene-dim">Step Type</label>
                                        <Select
                                          value={step.step_type}
                                          onValueChange={(v) => updateStep(index, { step_type: v })}
                                        >
                                          <SelectTrigger>
                                            <SelectValue />
                                          </SelectTrigger>
                                          <SelectContent>
                                            {STEP_TYPES.map((t) => (
                                              <SelectItem key={t} value={t}>
                                                {t}
                                              </SelectItem>
                                            ))}
                                          </SelectContent>
                                        </Select>
                                      </div>

                                      {/* Delay */}
                                      <div>
                                        <label className="text-xs font-medium text-browgene-dim">Delay (ms)</label>
                                        <Input
                                          type="number"
                                          value={step.delay_ms}
                                          onChange={(e) =>
                                            updateStep(index, { delay_ms: parseInt(e.target.value) || 0 })
                                          }
                                        />
                                      </div>

                                      {/* Description */}
                                      <div>
                                        <label className="text-xs font-medium text-browgene-dim">Description</label>
                                        <Input
                                          value={step.description}
                                          onChange={(e) => updateStep(index, { description: e.target.value })}
                                        />
                                      </div>

                                      {/* Timeout */}
                                      <div>
                                        <label className="text-xs font-medium text-browgene-dim">Timeout (ms)</label>
                                        <Input
                                          type="number"
                                          value={step.timeout_ms}
                                          onChange={(e) =>
                                            updateStep(index, { timeout_ms: parseInt(e.target.value) || 30000 })
                                          }
                                        />
                                      </div>

                                      {/* Max Retries */}
                                      <div>
                                        <label className="text-xs font-medium text-browgene-dim">Max Retries</label>
                                        <Input
                                          type="number"
                                          value={step.max_retries}
                                          onChange={(e) =>
                                            updateStep(index, { max_retries: parseInt(e.target.value) || 1 })
                                          }
                                          min={0}
                                          max={10}
                                        />
                                      </div>

                                      {/* On Failure */}
                                      <div>
                                        <label className="text-xs font-medium text-browgene-dim">On Failure</label>
                                        <Select
                                          value={step.on_failure}
                                          onValueChange={(v) => updateStep(index, { on_failure: v })}
                                        >
                                          <SelectTrigger>
                                            <SelectValue />
                                          </SelectTrigger>
                                          <SelectContent>
                                            <SelectItem value="stop">Stop</SelectItem>
                                            <SelectItem value="continue">Continue</SelectItem>
                                            <SelectItem value="ai_fallback">AI Fallback</SelectItem>
                                          </SelectContent>
                                        </Select>
                                      </div>

                                      {/* ── Type-specific params ─── */}

                                      {/* Navigate */}
                                      {step.step_type === "navigate" && (
                                        <div className="col-span-full">
                                          <label className="text-xs font-medium text-browgene-dim">URL</label>
                                          <Input
                                            value={String(params.url || "")}
                                            onChange={(e) => updateStepParams(index, { url: e.target.value })}
                                            placeholder="https://..."
                                          />
                                        </div>
                                      )}

                                      {/* Click */}
                                      {(step.step_type === "click" || step.step_type === "double_click" || step.step_type === "hover") && (
                                        <>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Element Text</label>
                                            <Input
                                              value={String(params.element_text || "")}
                                              onChange={(e) => updateStepParams(index, { element_text: e.target.value })}
                                              placeholder='e.g. "Sign In"'
                                            />
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Element Tag</label>
                                            <Select
                                              value={String(params.element_tag || "button")}
                                              onValueChange={(v) => updateStepParams(index, { element_tag: v })}
                                            >
                                              <SelectTrigger>
                                                <SelectValue />
                                              </SelectTrigger>
                                              <SelectContent>
                                                <SelectItem value="button">button</SelectItem>
                                                <SelectItem value="a">a (link)</SelectItem>
                                                <SelectItem value="div">div</SelectItem>
                                                <SelectItem value="span">span</SelectItem>
                                                <SelectItem value="input">input</SelectItem>
                                                <SelectItem value="select">select</SelectItem>
                                                <SelectItem value="li">li</SelectItem>
                                              </SelectContent>
                                            </Select>
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Selector (CSS)</label>
                                            <Input
                                              value={String(params.selector || "")}
                                              onChange={(e) => updateStepParams(index, { selector: e.target.value })}
                                              placeholder="Optional CSS selector"
                                            />
                                          </div>
                                        </>
                                      )}

                                      {/* Fill / Type */}
                                      {(step.step_type === "fill" || step.step_type === "type") && (
                                        <>
                                          <div className="col-span-full">
                                            <label className="text-xs font-medium text-browgene-dim">Value</label>
                                            <Input
                                              value={String(params.value || "")}
                                              onChange={(e) => updateStepParams(index, { value: e.target.value })}
                                              placeholder="Text to type"
                                            />
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Element Tag</label>
                                            <Input
                                              value={String(params.element_tag || "input")}
                                              onChange={(e) => updateStepParams(index, { element_tag: e.target.value })}
                                            />
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Element Text / Label</label>
                                            <Input
                                              value={String(params.element_text || "")}
                                              onChange={(e) => updateStepParams(index, { element_text: e.target.value })}
                                              placeholder="e.g. Email Address"
                                            />
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Selector (CSS)</label>
                                            <Input
                                              value={String(params.selector || "")}
                                              onChange={(e) => updateStepParams(index, { selector: e.target.value })}
                                              placeholder="Optional CSS selector"
                                            />
                                          </div>
                                        </>
                                      )}

                                      {/* Select dropdown */}
                                      {step.step_type === "select" && (
                                        <>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Option Text</label>
                                            <Input
                                              value={String(params.text || "")}
                                              onChange={(e) => updateStepParams(index, { text: e.target.value })}
                                              placeholder='e.g. "Lapsed"'
                                            />
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Element Tag</label>
                                            <Input
                                              value={String(params.element_tag || "select")}
                                              onChange={(e) => updateStepParams(index, { element_tag: e.target.value })}
                                            />
                                          </div>
                                          <div>
                                            <label className="text-xs font-medium text-browgene-dim">Selector (CSS)</label>
                                            <Input
                                              value={String(params.selector || "")}
                                              onChange={(e) => updateStepParams(index, { selector: e.target.value })}
                                              placeholder="Optional CSS selector"
                                            />
                                          </div>
                                        </>
                                      )}

                                      {/* Wait */}
                                      {step.step_type === "wait" && (
                                        <div>
                                          <label className="text-xs font-medium text-browgene-dim">Duration (seconds)</label>
                                          <Input
                                            type="number"
                                            step="0.5"
                                            value={String(params.seconds || 1)}
                                            onChange={(e) => updateStepParams(index, { seconds: parseFloat(e.target.value) })}
                                          />
                                        </div>
                                      )}

                                      {/* Extract */}
                                      {step.step_type === "extract" && (
                                        <div className="col-span-full">
                                          <label className="text-xs font-medium text-browgene-dim">Query / JS Expression</label>
                                          <Textarea
                                            value={String(params.query || params.js_expression || "")}
                                            onChange={(e) => updateStepParams(index, { query: e.target.value })}
                                            placeholder="What to extract from the page"
                                            className="min-h-[60px]"
                                          />
                                        </div>
                                      )}

                                      {/* Keyboard */}
                                      {step.step_type === "keyboard" && (
                                        <div>
                                          <label className="text-xs font-medium text-browgene-dim">Key</label>
                                          <Select
                                            value={String(params.key || "Enter")}
                                            onValueChange={(v) => updateStepParams(index, { key: v })}
                                          >
                                            <SelectTrigger>
                                              <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                              <SelectItem value="Enter">Enter</SelectItem>
                                              <SelectItem value="Tab">Tab</SelectItem>
                                              <SelectItem value="Escape">Escape</SelectItem>
                                              <SelectItem value="Backspace">Backspace</SelectItem>
                                              <SelectItem value="ArrowDown">Arrow Down</SelectItem>
                                              <SelectItem value="ArrowUp">Arrow Up</SelectItem>
                                              <SelectItem value="Space">Space</SelectItem>
                                            </SelectContent>
                                          </Select>
                                        </div>
                                      )}

                                      {/* Done */}
                                      {step.step_type === "done" && (
                                        <div className="col-span-full">
                                          <label className="text-xs font-medium text-browgene-dim">Result Text</label>
                                          <Textarea
                                            value={String(params.text || "")}
                                            onChange={(e) => updateStepParams(index, { text: e.target.value })}
                                            className="min-h-[80px] font-mono text-xs"
                                            placeholder="Final result text"
                                          />
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              )}

                              {/* Add step buttons */}
                              <div className="flex justify-center py-1.5 border-t border-border/10 bg-card/30">
                                <div className="flex items-center gap-1.5">
                                  <span className="text-[10px] text-muted-foreground mr-1">Add:</span>
                                  {[
                                    { type: "wait", icon: <Clock className="h-3 w-3 mr-0.5" />, label: "Wait" },
                                    { type: "click", icon: <MousePointer className="h-3 w-3 mr-0.5" />, label: "Click" },
                                    { type: "fill", icon: <Keyboard className="h-3 w-3 mr-0.5" />, label: "Fill" },
                                    { type: "navigate", icon: <Navigation className="h-3 w-3 mr-0.5" />, label: "Nav" },
                                    { type: "select", icon: <Filter className="h-3 w-3 mr-0.5" />, label: "Select" },
                                    { type: "extract", icon: <Download className="h-3 w-3 mr-0.5" />, label: "Extract" },
                                  ].map(({ type, icon, label }) => (
                                    <Button
                                      key={type}
                                      variant="ghost"
                                      size="sm"
                                      className="h-6 text-[10px] px-2"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        addStep(index, type);
                                      }}
                                    >
                                      {icon}
                                      {label}
                                    </Button>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}
                        </SortableStepItem>
                      );
                    })}
                  </div>
                </SortableContext>
              </DndContext>

              {/* Add first step if empty */}
              {task.steps.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <p className="mb-3">No steps yet. Add one to get started.</p>
                  <div className="flex justify-center gap-2">
                    <Button variant="outline" size="sm" onClick={() => addStep(-1, "navigate")}>
                      <Navigation className="h-4 w-4 mr-1" /> Navigate
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => addStep(-1, "click")}>
                      <MousePointer className="h-4 w-4 mr-1" /> Click
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => addStep(-1, "wait")}>
                      <Clock className="h-4 w-4 mr-1" /> Wait
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
        )}

        {/* Delete Step Confirmation */}
        <AlertDialog open={deleteStepIndex !== null} onOpenChange={(open) => !open && setDeleteStepIndex(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                Delete Step
              </AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete step {deleteStepIndex !== null ? deleteStepIndex + 1 : ""}? Remember to save after deleting.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                onClick={deleteStep}
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Execution Result Dialog */}
        <Dialog open={!!executionResult} onOpenChange={(open) => !open && setExecutionResult(null)}>
          <DialogContent className="sm:max-w-2xl max-h-[80vh] flex flex-col">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                {executionResult?.type === "success" ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : (
                  <XCircle className="h-5 w-5 text-destructive" />
                )}
                {executionResult?.title}
              </DialogTitle>
            </DialogHeader>
            <div className="overflow-y-auto max-h-[50vh] rounded-lg bg-secondary/20 p-4 text-sm whitespace-pre-line leading-relaxed">
              {executionResult?.message}
            </div>
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
      </main>
    </div>
  );
}
