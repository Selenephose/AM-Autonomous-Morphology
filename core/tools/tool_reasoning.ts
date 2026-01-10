/**
 * Tool-Use Reasoning Engine
 *
 * Enables:
 * - Tool discovery and registration
 * - Tool selection based on context
 * - Tool composition and chaining
 * - Permission-gated execution
 */

import * as tf from "@tensorflow/tfjs-node";

// ==================== TYPES ====================

export interface Tool {
  id: string;
  name: string;
  description: string;
  inputSchema: ToolSchema;
  outputSchema: ToolSchema;
  permissions: ToolPermission[];
  reliability: number;
  avgExecutionTime: number;
  usageCount: number;
  lastUsed: number;
  executor: ToolExecutor;
}

export interface ToolSchema {
  type: string;
  properties: Record<string, { type: string; description: string; required?: boolean }>;
}

export interface ToolPermission {
  resource: string;
  action: "read" | "write" | "execute" | "network" | "filesystem";
  grantedBy: string;
  expiresAt?: number;
}

export type ToolExecutor = (input: any, context: ToolContext) => Promise<ToolResult>;

export interface ToolContext {
  agentId: string;
  domain: string;
  permissions: ToolPermission[];
  timeout: number;
  metadata?: Record<string, unknown>;
}

export interface ToolResult {
  success: boolean;
  output: any;
  error?: string;
  executionTime: number;
  resourcesUsed?: string[];
}

export interface ToolChain {
  id: string;
  name: string;
  steps: ToolChainStep[];
  totalReliability: number;
  estimatedTime: number;
}

export interface ToolChainStep {
  toolId: string;
  inputMapping: Record<string, string>;
  outputKey: string;
  fallbackToolId?: string;
}

export interface ToolSelectionResult {
  selectedTool: Tool;
  confidence: number;
  reasoning: string[];
  alternatives: Array<{ tool: Tool; score: number }>;
}

// ==================== TOOL REGISTRY ====================

export class ToolRegistry {
  private tools: Map<string, Tool> = new Map();
  private categoryIndex: Map<string, string[]> = new Map();
  private capabilityIndex: Map<string, string[]> = new Map();

  register(tool: Omit<Tool, "usageCount" | "lastUsed">): string {
    const fullTool: Tool = {
      ...tool,
      usageCount: 0,
      lastUsed: 0,
    };

    this.tools.set(tool.id, fullTool);

    const category = this.inferCategory(tool);
    const categoryTools = this.categoryIndex.get(category) || [];
    categoryTools.push(tool.id);
    this.categoryIndex.set(category, categoryTools);

    const capabilities = this.inferCapabilities(tool);
    for (const cap of capabilities) {
      const capTools = this.capabilityIndex.get(cap) || [];
      capTools.push(tool.id);
      this.capabilityIndex.set(cap, capTools);
    }

    return tool.id;
  }

  private inferCategory(tool: Omit<Tool, "usageCount" | "lastUsed">): string {
    const name = tool.name.toLowerCase();
    const desc = tool.description.toLowerCase();

    if (name.includes("search") || desc.includes("search")) return "search";
    if (name.includes("calc") || desc.includes("math")) return "computation";
    if (name.includes("api") || desc.includes("api")) return "api";
    if (name.includes("file") || desc.includes("file")) return "filesystem";
    if (name.includes("browser") || desc.includes("web")) return "web";

    return "general";
  }

  private inferCapabilities(tool: Omit<Tool, "usageCount" | "lastUsed">): string[] {
    const capabilities: string[] = [];
    const desc = tool.description.toLowerCase();

    if (desc.includes("read")) capabilities.push("read");
    if (desc.includes("write")) capabilities.push("write");
    if (desc.includes("search")) capabilities.push("search");
    if (desc.includes("transform")) capabilities.push("transform");
    if (desc.includes("compute")) capabilities.push("compute");

    return capabilities;
  }

  getTool(id: string): Tool | undefined {
    return this.tools.get(id);
  }

  getByCategory(category: string): Tool[] {
    const ids = this.categoryIndex.get(category) || [];
    return ids.map(id => this.tools.get(id)!).filter(Boolean);
  }

  getByCapability(capability: string): Tool[] {
    const ids = this.capabilityIndex.get(capability) || [];
    return ids.map(id => this.tools.get(id)!).filter(Boolean);
  }

  getAllTools(): Tool[] {
    return Array.from(this.tools.values());
  }

  recordUsage(toolId: string, success: boolean, executionTime: number): void {
    const tool = this.tools.get(toolId);
    if (!tool) return;

    tool.usageCount++;
    tool.lastUsed = Date.now();

    const alpha = 1 / tool.usageCount;
    tool.reliability = tool.reliability * (1 - alpha) + (success ? 1 : 0) * alpha;
    tool.avgExecutionTime = tool.avgExecutionTime * (1 - alpha) + executionTime * alpha;
  }
}

// ==================== PERMISSION MANAGER ====================

export class PermissionManager {
  private grantedPermissions: Map<string, ToolPermission[]> = new Map();
  private permissionLog: Array<{ action: string; resource: string; granted: boolean; timestamp: number }> = [];

  grant(agentId: string, permission: Omit<ToolPermission, "grantedBy">): void {
    const fullPermission: ToolPermission = {
      ...permission,
      grantedBy: "system",
    };

    const permissions = this.grantedPermissions.get(agentId) || [];
    permissions.push(fullPermission);
    this.grantedPermissions.set(agentId, permissions);
  }

  revoke(agentId: string, resource: string, action: string): void {
    const permissions = this.grantedPermissions.get(agentId) || [];
    const filtered = permissions.filter(p => !(p.resource === resource && p.action === action));
    this.grantedPermissions.set(agentId, filtered);
  }

  check(agentId: string, tool: Tool): { allowed: boolean; missingPermissions: string[] } {
    const agentPermissions = this.grantedPermissions.get(agentId) || [];
    const missing: string[] = [];

    for (const required of tool.permissions) {
      const hasPermission = agentPermissions.some(p =>
        p.resource === required.resource &&
        p.action === required.action &&
        (!p.expiresAt || p.expiresAt > Date.now())
      );

      if (!hasPermission) {
        missing.push(`${required.action}:${required.resource}`);
      }
    }

    const allowed = missing.length === 0;

    this.permissionLog.push({
      action: "check",
      resource: tool.id,
      granted: allowed,
      timestamp: Date.now(),
    });

    return { allowed, missingPermissions: missing };
  }

  getPermissions(agentId: string): ToolPermission[] {
    return this.grantedPermissions.get(agentId) || [];
  }

  getLog(): Array<{ action: string; resource: string; granted: boolean; timestamp: number }> {
    return this.permissionLog;
  }
}

// ==================== TOOL SELECTOR ====================

export class ToolSelector {
  private registry: ToolRegistry;
  private selectionModel: tf.LayersModel;
  private selectionHistory: Array<{ context: number[]; toolId: string; success: boolean }> = [];

  constructor(registry: ToolRegistry, inputDim = 64) {
    this.registry = registry;
    this.selectionModel = this.buildModel(inputDim);
  }

  private buildModel(inputDim: number): tf.LayersModel {
    const input = tf.input({ shape: [inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: 32, activation: "sigmoid" }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    return model;
  }

  select(
    context: number[],
    goal: string,
    availableTools: Tool[]
  ): ToolSelectionResult {
    const scores: Array<{ tool: Tool; score: number }> = [];

    for (const tool of availableTools) {
      const score = this.scoreTool(tool, context, goal);
      scores.push({ tool, score });
    }

    scores.sort((a, b) => b.score - a.score);

    const selected = scores[0];
    const alternatives = scores.slice(1, 4);

    return {
      selectedTool: selected.tool,
      confidence: selected.score,
      reasoning: this.generateReasoning(selected.tool, goal, selected.score),
      alternatives,
    };
  }

  private scoreTool(tool: Tool, context: number[], goal: string): number {
    let score = 0;

    score += tool.reliability * 0.3;

    const goalLower = goal.toLowerCase();
    const descLower = tool.description.toLowerCase();
    const words = goalLower.split(/\s+/);
    const matches = words.filter(w => descLower.includes(w)).length;
    score += (matches / words.length) * 0.4;

    score += Math.log(tool.usageCount + 1) * 0.01;

    const speedScore = 1 / (tool.avgExecutionTime + 1);
    score += speedScore * 0.2;

    return Math.min(score, 1.0);
  }

  private generateReasoning(tool: Tool, goal: string, score: number): string[] {
    return [
      `Selected ${tool.name} for goal: "${goal}"`,
      `Tool reliability: ${(tool.reliability * 100).toFixed(1)}%`,
      `Match confidence: ${(score * 100).toFixed(1)}%`,
      `Previous uses: ${tool.usageCount}`,
    ];
  }

  recordSelection(context: number[], toolId: string, success: boolean): void {
    this.selectionHistory.push({ context, toolId, success });

    if (this.selectionHistory.length > 1000) {
      this.selectionHistory = this.selectionHistory.slice(-500);
    }
  }

  async train(): Promise<void> {
    if (this.selectionHistory.length < 50) return;

    const successfulSelections = this.selectionHistory.filter(s => s.success);
    if (successfulSelections.length === 0) return;
  }
}

// ==================== TOOL CHAIN BUILDER ====================

export class ToolChainBuilder {
  private registry: ToolRegistry;
  private chains: Map<string, ToolChain> = new Map();

  constructor(registry: ToolRegistry) {
    this.registry = registry;
  }

  build(
    goal: string,
    availableTools: Tool[],
    maxSteps = 5
  ): ToolChain | null {
    const requiredCapabilities = this.inferRequiredCapabilities(goal);
    const steps: ToolChainStep[] = [];

    let remainingCapabilities = [...requiredCapabilities];

    for (let i = 0; i < maxSteps && remainingCapabilities.length > 0; i++) {
      const capability = remainingCapabilities[0];
      const tool = this.findToolForCapability(capability, availableTools);

      if (!tool) break;

      steps.push({
        toolId: tool.id,
        inputMapping: { input: i === 0 ? "initial" : `step_${i - 1}_output` },
        outputKey: `step_${i}_output`,
      });

      remainingCapabilities = remainingCapabilities.filter(c => c !== capability);
    }

    if (steps.length === 0) return null;

    const totalReliability = steps.reduce((r, s) => {
      const tool = this.registry.getTool(s.toolId);
      return r * (tool?.reliability || 0.5);
    }, 1);

    const estimatedTime = steps.reduce((t, s) => {
      const tool = this.registry.getTool(s.toolId);
      return t + (tool?.avgExecutionTime || 1000);
    }, 0);

    const chain: ToolChain = {
      id: `chain_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      name: `Chain for: ${goal.slice(0, 30)}`,
      steps,
      totalReliability,
      estimatedTime,
    };

    this.chains.set(chain.id, chain);
    return chain;
  }

  private inferRequiredCapabilities(goal: string): string[] {
    const capabilities: string[] = [];
    const goalLower = goal.toLowerCase();

    if (goalLower.includes("search") || goalLower.includes("find")) {
      capabilities.push("search");
    }
    if (goalLower.includes("read") || goalLower.includes("get")) {
      capabilities.push("read");
    }
    if (goalLower.includes("write") || goalLower.includes("save")) {
      capabilities.push("write");
    }
    if (goalLower.includes("calculate") || goalLower.includes("compute")) {
      capabilities.push("compute");
    }
    if (goalLower.includes("transform") || goalLower.includes("convert")) {
      capabilities.push("transform");
    }

    return capabilities.length > 0 ? capabilities : ["general"];
  }

  private findToolForCapability(capability: string, tools: Tool[]): Tool | null {
    const capabilityTools = tools.filter(t =>
      t.description.toLowerCase().includes(capability)
    );

    if (capabilityTools.length === 0) return tools[0] || null;

    return capabilityTools.sort((a, b) => b.reliability - a.reliability)[0];
  }

  getChain(id: string): ToolChain | undefined {
    return this.chains.get(id);
  }
}

// ==================== TOOL EXECUTOR ====================

export class ToolExecutorEngine {
  private registry: ToolRegistry;
  private permissions: PermissionManager;
  private executionLog: Array<{
    toolId: string;
    success: boolean;
    time: number;
    timestamp: number;
  }> = [];

  constructor(registry: ToolRegistry, permissions: PermissionManager) {
    this.registry = registry;
    this.permissions = permissions;
  }

  async execute(
    toolId: string,
    input: any,
    context: ToolContext
  ): Promise<ToolResult> {
    const tool = this.registry.getTool(toolId);

    if (!tool) {
      return {
        success: false,
        output: null,
        error: `Tool ${toolId} not found`,
        executionTime: 0,
      };
    }

    const permCheck = this.permissions.check(context.agentId, tool);
    if (!permCheck.allowed) {
      return {
        success: false,
        output: null,
        error: `Missing permissions: ${permCheck.missingPermissions.join(", ")}`,
        executionTime: 0,
      };
    }

    const startTime = Date.now();

    try {
      const result = await Promise.race([
        tool.executor(input, context),
        new Promise<ToolResult>((_, reject) =>
          setTimeout(() => reject(new Error("Timeout")), context.timeout)
        ),
      ]);

      const executionTime = Date.now() - startTime;

      this.registry.recordUsage(toolId, result.success, executionTime);
      this.executionLog.push({
        toolId,
        success: result.success,
        time: executionTime,
        timestamp: Date.now(),
      });

      return { ...result, executionTime };
    } catch (error) {
      const executionTime = Date.now() - startTime;

      this.registry.recordUsage(toolId, false, executionTime);
      this.executionLog.push({
        toolId,
        success: false,
        time: executionTime,
        timestamp: Date.now(),
      });

      return {
        success: false,
        output: null,
        error: error instanceof Error ? error.message : "Unknown error",
        executionTime,
      };
    }
  }

  async executeChain(
    chain: ToolChain,
    initialInput: any,
    context: ToolContext
  ): Promise<{ success: boolean; outputs: Record<string, any>; errors: string[] }> {
    const outputs: Record<string, any> = { initial: initialInput };
    const errors: string[] = [];

    for (const step of chain.steps) {
      const input: Record<string, any> = {};
      for (const [key, mapping] of Object.entries(step.inputMapping)) {
        input[key] = outputs[mapping];
      }

      const result = await this.execute(step.toolId, input, context);

      if (!result.success) {
        if (step.fallbackToolId) {
          const fallbackResult = await this.execute(step.fallbackToolId, input, context);
          if (fallbackResult.success) {
            outputs[step.outputKey] = fallbackResult.output;
            continue;
          }
        }

        errors.push(result.error || `Step ${step.toolId} failed`);
        return { success: false, outputs, errors };
      }

      outputs[step.outputKey] = result.output;
    }

    return { success: true, outputs, errors };
  }

  getExecutionLog(): Array<{
    toolId: string;
    success: boolean;
    time: number;
    timestamp: number;
  }> {
    return this.executionLog;
  }
}

// ==================== TOOL REASONING ENGINE ====================

export class ToolReasoningEngine {
  registry: ToolRegistry;
  permissions: PermissionManager;
  selector: ToolSelector;
  chainBuilder: ToolChainBuilder;
  executor: ToolExecutorEngine;

  constructor(inputDim = 64) {
    this.registry = new ToolRegistry();
    this.permissions = new PermissionManager();
    this.selector = new ToolSelector(this.registry, inputDim);
    this.chainBuilder = new ToolChainBuilder(this.registry);
    this.executor = new ToolExecutorEngine(this.registry, this.permissions);

    this.registerBuiltinTools();
  }

  private registerBuiltinTools(): void {
    this.registry.register({
      id: "calculator",
      name: "Calculator",
      description: "Performs mathematical calculations and computations",
      inputSchema: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression to evaluate", required: true },
        },
      },
      outputSchema: {
        type: "object",
        properties: {
          result: { type: "number", description: "Calculation result" },
        },
      },
      permissions: [],
      reliability: 0.99,
      avgExecutionTime: 10,
      executor: async (input: { expression: string }) => {
        try {
          const result = Function(`"use strict"; return (${input.expression})`)();
          return { success: true, output: { result }, executionTime: 5 };
        } catch (e) {
          return { success: false, output: null, error: "Invalid expression", executionTime: 5 };
        }
      },
    });

    this.registry.register({
      id: "text_transformer",
      name: "Text Transformer",
      description: "Transforms and processes text data",
      inputSchema: {
        type: "object",
        properties: {
          text: { type: "string", description: "Input text", required: true },
          operation: { type: "string", description: "Operation: uppercase, lowercase, reverse" },
        },
      },
      outputSchema: {
        type: "object",
        properties: {
          result: { type: "string", description: "Transformed text" },
        },
      },
      permissions: [],
      reliability: 0.95,
      avgExecutionTime: 20,
      executor: async (input: { text: string; operation: string }) => {
        let result = input.text;
        switch (input.operation) {
          case "uppercase":
            result = result.toUpperCase();
            break;
          case "lowercase":
            result = result.toLowerCase();
            break;
          case "reverse":
            result = result.split("").reverse().join("");
            break;
        }
        return { success: true, output: { result }, executionTime: 10 };
      },
    });
  }

  async reason(
    context: number[],
    goal: string,
    agentId: string
  ): Promise<{
    selectedTool: Tool | null;
    chain: ToolChain | null;
    reasoning: string[];
  }> {
    const tools = this.registry.getAllTools();

    const allowedTools = tools.filter(t =>
      this.permissions.check(agentId, t).allowed
    );

    if (allowedTools.length === 0) {
      return {
        selectedTool: null,
        chain: null,
        reasoning: ["No tools available with current permissions"],
      };
    }

    const selection = this.selector.select(context, goal, allowedTools);

    const chain = this.chainBuilder.build(goal, allowedTools);

    return {
      selectedTool: selection.selectedTool,
      chain,
      reasoning: selection.reasoning,
    };
  }

  async useTool(
    toolId: string,
    input: any,
    agentId: string,
    domain: string
  ): Promise<ToolResult> {
    const context: ToolContext = {
      agentId,
      domain,
      permissions: this.permissions.getPermissions(agentId),
      timeout: 30000,
    };

    const result = await this.executor.execute(toolId, input, context);

    this.selector.recordSelection([], toolId, result.success);

    return result;
  }

  grantPermission(agentId: string, resource: string, action: "read" | "write" | "execute" | "network" | "filesystem"): void {
    this.permissions.grant(agentId, { resource, action });
  }

  getStats(): Record<string, unknown> {
    return {
      registeredTools: this.registry.getAllTools().length,
      executionLog: this.executor.getExecutionLog().slice(-50),
    };
  }
}
