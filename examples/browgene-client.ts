/**
 * SVG-Browgene API Client
 * 
 * This TypeScript client demonstrates how to interact with the SVG-Browgene API
 * to run browser automation tasks and check their status.
 */

// API endpoint configuration
const API_CONFIG = {
  BASE_URL: 'http://localhost:7793',
  ENDPOINTS: {
    RUN_TASK: '/api/browgene/run_task',
    TASK_STATUS: '/api/browgene/task_status',
  },
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  }
};

// Task request interface
interface BrowgeneTaskRequest {
  task: string;
  agent_type?: string;
  llm_provider?: string;
  llm_model_name?: string;
  llm_num_ctx?: number;
  llm_temperature?: number;
  llm_base_url?: string;
  llm_api_key?: string;
  use_own_browser?: boolean;
  keep_browser_open?: boolean;
  headless?: boolean;
  disable_security?: boolean;
  window_w?: number;
  window_h?: number;
  save_recording_path?: string;
  save_agent_history_path?: string;
  save_trace_path?: string;
  enable_recording?: boolean;
  max_steps?: number;
  use_vision?: boolean;
  max_actions_per_step?: number;
  tool_calling_method?: string;
  add_infos?: string;
}

// Task response interface
interface BrowgeneTaskResponse {
  success: boolean;
  extracted_text: string;
  interactions: any[];
  agent_brain: Record<string, any>;
  history_file: string | null;
  error: string | null;
  metadata: {
    task_id: string;
    task: string;
    agent_type?: string;
    llm_provider?: string;
    llm_model_name?: string;
    status: string;
    start_time?: string;
    end_time?: string;
  };
}

/**
 * SVG-Browgene API Client class
 */
class BrowgeneClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  /**
   * Create a new Browgene API client
   * @param baseUrl Base URL for the API (default: http://localhost:7793)
   * @param apiKey Optional API key for authentication
   */
  constructor(baseUrl: string = API_CONFIG.BASE_URL, apiKey?: string) {
    this.baseUrl = baseUrl;
    this.headers = { ...API_CONFIG.DEFAULT_HEADERS };
    
    if (apiKey) {
      this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
  }

  /**
   * Start a new browser automation task
   * @param taskRequest Task configuration
   * @returns Task response with task_id for status polling
   */
  async runTask(taskRequest: BrowgeneTaskRequest): Promise<BrowgeneTaskResponse> {
    const url = `${this.baseUrl}${API_CONFIG.ENDPOINTS.RUN_TASK}`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(taskRequest),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      return await response.json() as BrowgeneTaskResponse;
    } catch (error) {
      console.error('Error running task:', error);
      throw error;
    }
  }

  /**
   * Check the status of a running task
   * @param taskId Task ID returned from runTask
   * @returns Current task status and results if complete
   */
  async checkTaskStatus(taskId: string): Promise<BrowgeneTaskResponse> {
    const url = `${this.baseUrl}${API_CONFIG.ENDPOINTS.TASK_STATUS}/${taskId}`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: this.headers,
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      return await response.json() as BrowgeneTaskResponse;
    } catch (error) {
      console.error('Error checking task status:', error);
      throw error;
    }
  }

  /**
   * Run a task and poll for completion
   * @param taskRequest Task configuration
   * @param pollingIntervalMs Polling interval in milliseconds (default: 2000)
   * @param timeoutMs Maximum time to wait in milliseconds (default: 300000 - 5 minutes)
   * @returns Final task result
   */
  async runTaskAndWaitForCompletion(
    taskRequest: BrowgeneTaskRequest, 
    pollingIntervalMs: number = 2000,
    timeoutMs: number = 300000
  ): Promise<BrowgeneTaskResponse> {
    // Start the task
    const initialResponse = await this.runTask(taskRequest);
    const taskId = initialResponse.metadata.task_id;
    
    if (!taskId) {
      throw new Error('No task ID returned from API');
    }
    
    console.log(`Task started with ID: ${taskId}`);
    
    // Poll for completion
    const startTime = Date.now();
    let lastResponse = initialResponse;
    
    while (Date.now() - startTime < timeoutMs) {
      // Wait for the polling interval
      await new Promise(resolve => setTimeout(resolve, pollingIntervalMs));
      
      // Check status
      const statusResponse = await this.checkTaskStatus(taskId);
      lastResponse = statusResponse;
      
      console.log(`Task status: ${statusResponse.metadata.status}`);
      
      // If task is complete (success or failure), return the result
      if (['completed', 'failed'].includes(statusResponse.metadata.status)) {
        return statusResponse;
      }
    }
    
    throw new Error(`Task timed out after ${timeoutMs}ms`);
  }
}

/**
 * Example usage of the Browgene client
 */
async function exampleUsage() {
  // Create a client instance
  const client = new BrowgeneClient();
  
  // Define a task
  const task: BrowgeneTaskRequest = {
    task: "Go to cnn.com and get the top headline",
    agent_type: "org",
    llm_provider: "openai",
    llm_model_name: "gpt-4o",
    llm_temperature: 0.7,
    headless: true,
    disable_security: false,
    window_w: 1280,
    window_h: 720,
    enable_recording: false,
    max_steps: 50,
    use_vision: true,
    max_actions_per_step: 10
  };
  
  try {
    // Method 1: Start task and manually poll for status
    console.log("Method 1: Manual polling");
    const response = await client.runTask(task);
    console.log(`Task started with ID: ${response.metadata.task_id}`);
    
    // Check status after 5 seconds
    setTimeout(async () => {
      const status = await client.checkTaskStatus(response.metadata.task_id);
      console.log("Task status:", status.metadata.status);
      console.log("Task result:", status.extracted_text);
    }, 5000);
    
    // Method 2: Run task and automatically wait for completion
    console.log("\nMethod 2: Automatic polling");
    const result = await client.runTaskAndWaitForCompletion(task);
    
    console.log("Task completed!");
    console.log("Success:", result.success);
    console.log("Extracted text:", result.extracted_text);
    console.log("Interactions:", result.interactions.length);
    
    if (result.error) {
      console.error("Error:", result.error);
    }
  } catch (error) {
    console.error("Error in example:", error);
  }
}

// Run the example if this file is executed directly
if (require.main === module) {
  exampleUsage().catch(console.error);
}

// Export the client for use in other files
export { BrowgeneClient, BrowgeneTaskRequest, BrowgeneTaskResponse };
