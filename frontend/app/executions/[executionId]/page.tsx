"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, CheckCircle2, XCircle, Clock, Loader2, Play } from "lucide-react";

interface StepResult {
  step_index: number;
  step_type: string;
  description: string;
  status: string;
  duration_ms: number;
  error: string | null;
  has_screenshot_before: boolean;
  has_screenshot_after: boolean;
  has_extracted_data: boolean;
}

interface ExecutionData {
  execution_id: string;
  task_name: string;
  parameters: Record<string, string>;
  mode: string;
  status: string;
  steps_completed: number;
  steps_total: number;
  start_time: string;
  end_time: string | null;
  memory: Record<string, string>;
  screenshots: string[];
  extracted_data: string[];
  error: string | null;
  step_results: StepResult[];
}

export default function ExecutionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const executionId = params.executionId as string;

  const [execution, setExecution] = useState<ExecutionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExecution = async () => {
      try {
        const res = await fetch(`/api/browgene/executions/${executionId}`);
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        const data = await res.json();
        setExecution(data);
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    };

    fetchExecution();

    // Poll if still running
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/browgene/executions/${executionId}`);
        if (!res.ok) return;
        const data = await res.json();
        setExecution(data);
        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
        }
      } catch {
        // ignore polling errors
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [executionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-cyan-400" />
      </div>
    );
  }

  if (error || !execution) {
    return (
      <div className="max-w-4xl mx-auto p-8">
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400">
          {error || "Execution not found"}
        </div>
      </div>
    );
  }

  const statusIcon = {
    completed: <CheckCircle2 className="h-5 w-5 text-green-400" />,
    failed: <XCircle className="h-5 w-5 text-red-400" />,
    running: <Loader2 className="h-5 w-5 animate-spin text-cyan-400" />,
    pending: <Clock className="h-5 w-5 text-yellow-400" />,
  }[execution.status] || <Clock className="h-5 w-5 text-gray-400" />;

  const statusColor = {
    completed: "text-green-400",
    failed: "text-red-400",
    running: "text-cyan-400",
    pending: "text-yellow-400",
  }[execution.status] || "text-gray-400";

  const stepTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      navigate: "🧭",
      click: "👆",
      fill: "✏️",
      extract: "📋",
      done: "✅",
      scroll: "📜",
      keyboard: "⌨️",
      wait: "⏳",
      ai_act: "🤖",
      select: "📝",
    };
    return icons[type] || "▶️";
  };

  return (
    <div className="max-w-4xl mx-auto p-8 space-y-6">
      {/* Back button */}
      <button
        onClick={() => router.back()}
        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </button>

      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          {statusIcon}
          <h1 className="text-2xl font-mono font-bold tracking-tight">
            Execution {execution.execution_id}
          </h1>
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>Task: <span className="text-cyan-400 font-mono">{execution.task_name}</span></span>
          <span className={statusColor}>{execution.status}</span>
          <span>{execution.steps_completed}/{execution.steps_total} steps</span>
        </div>
      </div>

      {/* Timing */}
      <div className="rounded-lg border border-border/50 bg-secondary/20 p-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-muted-foreground">Started: </span>
          <span className="font-mono">{execution.start_time}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Ended: </span>
          <span className="font-mono">{execution.end_time || "—"}</span>
        </div>
      </div>

      {/* Error */}
      {execution.error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400 text-sm font-mono">
          {execution.error}
        </div>
      )}

      {/* Steps */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Play className="h-4 w-4 text-cyan-400" />
          Step Results ({execution.step_results.length})
        </h2>

        <div className="space-y-2">
          {execution.step_results.map((step) => (
            <div
              key={step.step_index}
              className="rounded-lg border border-border/50 bg-secondary/20 p-3 flex items-center gap-3"
            >
              <span className="text-lg">{stepTypeIcon(step.step_type)}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-muted-foreground">
                    #{step.step_index + 1}
                  </span>
                  <span className="text-sm font-medium">{step.step_type}</span>
                  <span className="text-xs text-muted-foreground truncate">
                    {step.description}
                  </span>
                </div>
                {step.error && (
                  <p className="text-xs text-red-400 mt-1">{step.error}</p>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="text-xs text-muted-foreground">
                  {step.duration_ms}ms
                </span>
                {step.status === "success" ? (
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                ) : step.status === "failed" ? (
                  <XCircle className="h-4 w-4 text-red-400" />
                ) : (
                  <Clock className="h-4 w-4 text-yellow-400" />
                )}
              </div>
            </div>
          ))}

          {execution.step_results.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-8">
              No step results yet.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
