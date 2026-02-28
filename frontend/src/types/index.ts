export interface BrowserStep {
  step_type: string;
  params: Record<string, unknown>;
  description: string;
  delay_ms: number;
  take_screenshot: boolean;
  on_failure: string;
  max_retries: number;
  timeout_ms: number;
  status?: string;
  screenshot_before?: string;
  screenshot_after?: string;
  extracted_data?: unknown;
  error?: string;
  duration_ms?: number;
}

export type TaskMode = "deterministic" | "agentic";

export interface BrowserTask {
  name: string;
  description: string;
  steps: BrowserStep[];
  parameters: Record<string, string>;
  created_at: string;
  updated_at: string;
  source: string;
  tags: string[];
  metadata: Record<string, unknown>;
  start_url?: string;
  requires_auth: boolean;
  auth_credential_key?: string;
  mode: TaskMode;
  goal?: string;
  max_agent_steps?: number;
}

export interface TaskListItem {
  name: string;
  description: string;
  steps: string;
  source: string;
  tags: string;
  start_url: string;
  created_at: string;
  updated_at: string;
  mode?: TaskMode;
  goal?: string;
}

export interface TaskExecution {
  execution_id: string;
  task_name: string;
  parameters: Record<string, unknown>;
  mode: string;
  status: string;
  steps_completed: number;
  steps_total: number;
  start_time?: string;
  end_time?: string;
  memory: Record<string, unknown>;
  screenshots: string[];
  extracted_data: unknown[];
  error?: string;
  step_results: StepResult[];
}

export interface StepResult {
  step_index: number;
  step_type: string;
  description: string;
  status: string;
  duration_ms?: number;
  error?: string;
  has_screenshot_before: boolean;
  has_screenshot_after: boolean;
  has_extracted_data: boolean;
}

export interface ExplorationResult {
  exploration_id: string;
  task_description: string;
  recorded_actions: RecordedAction[];
  final_url: string;
  final_screenshot?: string;
  success: boolean;
  error?: string;
  status: "running" | "completed" | "failed";
  start_time: string;
  end_time?: string;
  agent_output?: string;
  action_count: number;
}

export interface RecordedAction {
  action_type: string;
  params: Record<string, unknown>;
  description: string;
  screenshot_before?: string;
  screenshot_after?: string;
  page_url: string;
  timestamp: string;
}

export interface BrowserStatus {
  active: boolean;
  url?: string;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  modes: string[];
  browser_active: boolean;
}
