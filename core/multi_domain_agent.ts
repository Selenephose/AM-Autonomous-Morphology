/**
 * Multi-Domain Learning Agent
 *
 * A general-purpose AGI-like agent that integrates:
 * - Hierarchical memory (episodic, semantic, skill, value)
 * - Dreamer-style world model with imagination
 * - Lifelong learning with transfer capabilities
 * - Multi-agent inner debate for reasoning
 * - Meta-cognitive self-reflection
 * - Autonomous skill discovery
 * - Curiosity-driven exploration
 * - Tool-use reasoning
 * - Long-horizon goal tracking
 * - Consciousness-like self-monitoring
 */

import * as tf from "@tensorflow/tfjs-node";

import { HierarchicalMemorySystem, Episode, EpisodeEvent, EpisodeOutcome } from "./memory";
import { DreamerWorldModel, WorldState } from "./world";
import { LifelongLearner, DomainKnowledge } from "./learning";
import { InnerDebateOrchestrator, DebateResult } from "./agents";
import { MetaCognitiveLoop, DomainCompetence } from "./reflection";
import { SkillDiscoveryEngine, DiscoveredSkill, BehaviorStep, BehaviorOutcome } from "./skills";
import { CuriosityEngine, NoveltySignal } from "./curiosity";
import { ToolReasoningEngine, Tool, ToolResult } from "./tools";
import { GoalTracker, Goal } from "./goals";
import { ConsciousnessMonitor, ConsciousnessState, IntrospectionReport } from "./consciousness";

// ==================== TYPES ====================

export interface AgentConfig {
  name: string;
  latentDim?: number;
  beliefDim?: number;
  actionDim?: number;
  inputDim?: number;
  maxEpisodes?: number;
  intrinsicRewardWeight?: number;
}

export interface AgentState {
  currentDomain: string | null;
  currentGoal: Goal | null;
  worldState: WorldState;
  consciousness: ConsciousnessState;
  activeSkills: string[];
  recentDecisions: string[];
}

export interface StepResult {
  action: number[];
  reasoning: string[];
  novelty: NoveltySignal;
  confidence: number;
  skillUsed?: string;
  toolUsed?: string;
}

export interface EpisodeResult {
  domain: string;
  totalReward: number;
  intrinsicReward: number;
  extrinsicReward: number;
  skillsDiscovered: DiscoveredSkill[];
  lessonsLearned: string[];
  goalsProgressed: string[];
}

export interface ReflectionResult {
  insights: string[];
  strategyChanges: string[];
  architectureProposals: string[];
  competenceUpdates: Map<string, number>;
}

// ==================== MULTI-DOMAIN AGENT ====================

export class MultiDomainAgent {
  // Configuration
  readonly name: string;
  private config: AgentConfig;

  // Core Systems
  memory: HierarchicalMemorySystem;
  worldModel: DreamerWorldModel;
  learner: LifelongLearner;
  debate: InnerDebateOrchestrator;
  reflection: MetaCognitiveLoop;
  skillDiscovery: SkillDiscoveryEngine;
  curiosity: CuriosityEngine;
  tools: ToolReasoningEngine;
  goals: GoalTracker;
  consciousness: ConsciousnessMonitor;

  // State
  private currentDomain: string | null = null;
  private currentEpisode: {
    steps: BehaviorStep[];
    context: number[];
    startTime: number;
  } | null = null;
  private episodeCount: number = 0;
  private totalSteps: number = 0;
  private previousSymbols: string[] = [];

  constructor(config: AgentConfig) {
    this.name = config.name;
    this.config = {
      latentDim: 32,
      beliefDim: 256,
      actionDim: 4,
      inputDim: 64,
      maxEpisodes: 10000,
      intrinsicRewardWeight: 0.5,
      ...config,
    };

    this.memory = new HierarchicalMemorySystem({
      maxEpisodes: this.config.maxEpisodes,
      latentDim: this.config.latentDim,
    });

    this.worldModel = new DreamerWorldModel(
      this.config.latentDim!,
      this.config.beliefDim!,
      this.config.actionDim!
    );

    this.learner = new LifelongLearner({
      inputDim: this.config.inputDim!,
      hiddenDim: 128,
      outputDim: this.config.actionDim!,
    });

    this.debate = new InnerDebateOrchestrator(this.config.inputDim!);

    this.reflection = new MetaCognitiveLoop(this.config.inputDim!);

    this.skillDiscovery = new SkillDiscoveryEngine({
      maxBehaviors: 10000,
      minClusterSize: 3,
      similarityThreshold: 0.7,
    });

    this.curiosity = new CuriosityEngine(this.config.inputDim!);

    this.tools = new ToolReasoningEngine(this.config.inputDim!);

    this.goals = new GoalTracker();

    this.consciousness = new ConsciousnessMonitor();

    this.consciousness.metacognition.updateSelfModel(
      ["multi-domain learning", "world modeling", "skill discovery", "tool use", "goal tracking"],
      ["requires training data", "bounded computational resources"],
      ["exploration-exploitation balance", "curiosity-driven"]
    );
  }

  // ==================== DOMAIN MANAGEMENT ====================

  async enterDomain(domainName: string, sampleData?: number[][]): Promise<string> {
    this.consciousness.attention.setFocus(`domain:${domainName}`);
    this.consciousness.motivation.recordActivity("exploration");

    const domainId = await this.learner.startDomain(domainName, sampleData);

    this.currentDomain = domainId;

    this.tools.grantPermission(this.name, domainName, "read");
    this.tools.grantPermission(this.name, domainName, "execute");

    return domainId;
  }

  getCurrentDomain(): string | null {
    return this.currentDomain;
  }

  // ==================== EPISODE MANAGEMENT ====================

  startEpisode(context: number[]): void {
    this.currentEpisode = {
      steps: [],
      context,
      startTime: Date.now(),
    };

    this.episodeCount++;
    this.worldModel.reset();
    this.previousSymbols = [];

    this.consciousness.attention.setFocus("episode_start");
  }

  async step(
    observation: number[],
    extrinsicReward: number,
    done: boolean
  ): Promise<StepResult> {
    this.totalSteps++;
    this.consciousness.attention.setFocus("step_execution");

    const worldState = await this.worldModel.observe(
      observation,
      Array(this.config.actionDim!).fill(0),
      extrinsicReward,
      done
    );

    const prediction = await this.worldModel.predictFuture([Array(this.config.actionDim!).fill(0)]);
    const predictedState = prediction.states[1]?.latent || worldState.latent;

    const currentSymbol = this.assignSymbol(worldState.latent);
    const novelty = this.curiosity.computeIntrinsicReward(
      observation,
      predictedState,
      worldState.latent,
      [currentSymbol],
      this.previousSymbols,
      this.getHierarchyDepth()
    );

    const totalReward = extrinsicReward + novelty.totalIntrinsic * this.config.intrinsicRewardWeight!;

    const debateResult = await this.debate.debate(observation, { reward: totalReward });

    const action = this.extractAction(debateResult);
    const confidence = debateResult.consensus;

    const decisionId = await this.reflection.recordDecision(
      this.currentDomain || "unknown",
      observation,
      action,
      debateResult.winningProposal.reasoning
    );

    const applicableSkills = this.skillDiscovery.getApplicableSkills(observation, this.currentDomain || undefined);
    let skillUsed: string | undefined;

    if (applicableSkills.length > 0 && Math.random() < 0.3) {
      const skill = applicableSkills[0];
      skillUsed = skill.id;
      this.skillDiscovery.useSkill(skill.id, totalReward > 0, totalReward);
    }

    const toolReasoning = await this.tools.reason(
      observation,
      "optimize_action",
      this.name
    );
    let toolUsed: string | undefined;

    if (toolReasoning.selectedTool && Math.random() < 0.2) {
      toolUsed = toolReasoning.selectedTool.id;
    }

    if (this.currentEpisode) {
      this.currentEpisode.steps.push({
        state: observation,
        action,
        latent: worldState.latent,
        symbol: currentSymbol,
        reward: totalReward,
      });
    }

    this.consciousness.confidence.update(
      this.currentDomain || "unknown",
      decisionId,
      confidence,
      totalReward > 0 ? 1 : 0
    );

    this.consciousness.uncertainty.update(
      worldState.uncertainty,
      novelty.predictionError,
      "step_uncertainty"
    );

    this.previousSymbols.push(currentSymbol);
    if (this.previousSymbols.length > 20) {
      this.previousSymbols = this.previousSymbols.slice(-10);
    }

    return {
      action,
      reasoning: debateResult.winningProposal.reasoning,
      novelty,
      confidence,
      skillUsed,
      toolUsed,
    };
  }

  async endEpisode(): Promise<EpisodeResult> {
    if (!this.currentEpisode) {
      return this.createEmptyEpisodeResult();
    }

    const duration = Date.now() - this.currentEpisode.startTime;
    const totalReward = this.currentEpisode.steps.reduce((s, step) => s + step.reward, 0);
    const extrinsicReward = totalReward * (1 - this.config.intrinsicRewardWeight!);
    const intrinsicReward = totalReward * this.config.intrinsicRewardWeight!;

    const episode: Omit<Episode, "id" | "accessCount" | "lastAccessed"> = {
      timestamp: this.currentEpisode.startTime,
      domain: this.currentDomain || "unknown",
      context: this.currentEpisode.context,
      events: this.currentEpisode.steps.map((step, i) => ({
        step: i,
        state: step.state,
        action: step.action,
        latent: step.latent,
        symbol: step.symbol,
        reward: step.reward,
      })),
      outcome: {
        success: totalReward > 0,
        totalReward,
        goalAchieved: false,
        lessonsLearned: this.extractLessons(this.currentEpisode.steps),
      },
      importance: Math.abs(totalReward) / 10,
    };

    this.memory.episodic.store(episode);

    const behaviorOutcome: BehaviorOutcome = {
      totalReward,
      success: totalReward > 0,
      goalReached: false,
      duration,
    };

    this.skillDiscovery.recordBehavior(
      this.currentEpisode.steps,
      this.currentEpisode.context,
      behaviorOutcome,
      this.currentDomain || "unknown"
    );

    const skillsDiscovered = await this.skillDiscovery.discover(this.currentDomain || undefined);

    await this.worldModel.train();

    await this.curiosity.train(this.currentEpisode.steps.map(s => s.state));
    const newConcepts = this.curiosity.formConcepts();

    for (const concept of newConcepts) {
      this.memory.semantic.store({
        name: concept.id,
        embedding: concept.embedding,
        relatedConcepts: [],
        sourceEpisodes: [],
        confidence: concept.coherence,
        domain: this.currentDomain || "unknown",
        abstractionLevel: 1,
      });
    }

    this.consciousness.competence.update(
      this.currentDomain || "unknown",
      totalReward > 0 ? 0.7 : 0.3
    );

    const goalsProgressed: string[] = [];
    const activeGoal = this.goals.getActiveGoal();
    if (activeGoal && totalReward > 0) {
      const currentProgress = activeGoal.progress;
      const newProgress = Math.min(100, currentProgress + 5);
      this.goals.updateProgress(activeGoal.id, newProgress, "Episode completion");
      goalsProgressed.push(activeGoal.id);
    }

    this.curiosity.decay();

    this.currentEpisode = null;

    return {
      domain: this.currentDomain || "unknown",
      totalReward,
      intrinsicReward,
      extrinsicReward,
      skillsDiscovered,
      lessonsLearned: episode.outcome.lessonsLearned,
      goalsProgressed,
    };
  }

  private createEmptyEpisodeResult(): EpisodeResult {
    return {
      domain: "unknown",
      totalReward: 0,
      intrinsicReward: 0,
      extrinsicReward: 0,
      skillsDiscovered: [],
      lessonsLearned: [],
      goalsProgressed: [],
    };
  }

  private extractLessons(steps: BehaviorStep[]): string[] {
    const lessons: string[] = [];

    if (steps.length === 0) return lessons;

    const avgReward = steps.reduce((s, step) => s + step.reward, 0) / steps.length;
    if (avgReward > 0.5) {
      lessons.push("Positive reward trajectory - reinforce similar patterns");
    } else if (avgReward < -0.5) {
      lessons.push("Negative reward trajectory - avoid similar patterns");
    }

    const uniqueSymbols = new Set(steps.map(s => s.symbol)).size;
    if (uniqueSymbols > steps.length * 0.8) {
      lessons.push("High symbol diversity - explore more structured patterns");
    }

    const rewardTrend = steps.slice(-5).reduce((s, step) => s + step.reward, 0) -
      steps.slice(0, 5).reduce((s, step) => s + step.reward, 0);
    if (rewardTrend > 1) {
      lessons.push("Improving reward trend - strategy is working");
    } else if (rewardTrend < -1) {
      lessons.push("Declining reward trend - consider strategy change");
    }

    return lessons;
  }

  // ==================== REFLECTION & INTROSPECTION ====================

  async reflect(): Promise<ReflectionResult> {
    this.consciousness.attention.setFocus("reflection");
    this.consciousness.motivation.recordActivity("learning");

    const reflectionResult = await this.reflection.reflect();

    const insights: string[] = [];
    const strategyChanges: string[] = [];
    const architectureProposals: string[] = [];

    for (const pattern of reflectionResult.patterns) {
      insights.push(`Failure pattern: ${pattern.pattern} (${pattern.occurrences} occurrences)`);
    }

    for (const adjustment of reflectionResult.adjustments) {
      strategyChanges.push(`${adjustment.domain}: ${adjustment.previousStrategy} -> ${adjustment.newStrategy}`);
    }

    if (reflectionResult.proposals) {
      architectureProposals.push(reflectionResult.proposals.proposedChange);
    }

    const competenceUpdates = new Map<string, number>();
    for (const comp of reflectionResult.competence) {
      competenceUpdates.set(comp.domain, comp.overallScore);
    }

    if (this.consciousness.shouldIntrospect()) {
      const introspection = this.consciousness.introspect();
      insights.push(...introspection.insights);
    }

    return {
      insights,
      strategyChanges,
      architectureProposals,
      competenceUpdates,
    };
  }

  introspect(): IntrospectionReport {
    return this.consciousness.introspect();
  }

  // ==================== GOAL MANAGEMENT ====================

  createGoal(
    name: string,
    description: string,
    estimatedEffort: number,
    options: { priority?: number; deadline?: number } = {}
  ): string {
    const goalId = this.goals.createGoal(
      name,
      description,
      this.currentDomain || "general",
      estimatedEffort,
      options
    );

    this.consciousness.motivation.recordActivity("goal");

    return goalId;
  }

  activateGoal(goalId: string): boolean {
    const result = this.goals.activateGoal(goalId);
    if (result) {
      this.consciousness.attention.setFocus(`goal:${goalId}`);
    }
    return result;
  }

  decomposeGoal(goalId: string): string[] {
    return this.goals.decomposeGoal(goalId);
  }

  getActiveGoal(): Goal | null {
    return this.goals.getActiveGoal();
  }

  // ==================== SKILL MANAGEMENT ====================

  getApplicableSkills(state: number[]): DiscoveredSkill[] {
    return this.skillDiscovery.getApplicableSkills(state, this.currentDomain || undefined);
  }

  async discoverSkills(): Promise<DiscoveredSkill[]> {
    return this.skillDiscovery.discover(this.currentDomain || undefined);
  }

  // ==================== TOOL USE ====================

  async useTool(toolId: string, input: any): Promise<ToolResult> {
    this.consciousness.attention.setFocus(`tool:${toolId}`);

    const result = await this.tools.useTool(
      toolId,
      input,
      this.name,
      this.currentDomain || "general"
    );

    this.consciousness.confidence.update(
      "tool_use",
      toolId,
      result.success ? 0.9 : 0.3,
      result.success ? 1 : 0
    );

    return result;
  }

  getAvailableTools(): Tool[] {
    return this.tools.registry.getAllTools();
  }

  // ==================== IMAGINATION & PLANNING ====================

  async imagine(
    actionSequence: number[][],
    fromState?: WorldState
  ): Promise<{ states: WorldState[]; totalReward: number }> {
    return this.worldModel.predictFuture(actionSequence, fromState);
  }

  async counterfactual(
    hypotheticalAction: number[],
    alternativeAction: number[]
  ): Promise<{
    recommendation: "factual" | "counterfactual" | "uncertain";
    difference: number;
  }> {
    const currentState = this.worldModel.getCurrentState();

    const result = await this.worldModel.counterfactual.whatIf(
      currentState,
      hypotheticalAction,
      alternativeAction
    );

    return {
      recommendation: result.recommendation,
      difference: result.difference,
    };
  }

  // ==================== STATE ACCESS ====================

  getState(): AgentState {
    return {
      currentDomain: this.currentDomain,
      currentGoal: this.goals.getActiveGoal(),
      worldState: this.worldModel.getCurrentState(),
      consciousness: this.consciousness.getState(),
      activeSkills: [],
      recentDecisions: [],
    };
  }

  getConsciousnessState(): ConsciousnessState {
    return this.consciousness.getState();
  }

  getStats(): Record<string, unknown> {
    return {
      name: this.name,
      episodeCount: this.episodeCount,
      totalSteps: this.totalSteps,
      currentDomain: this.currentDomain,
      memory: this.memory.getFullStats(),
      worldModel: this.worldModel.getStats(),
      learner: this.learner.getProgress(),
      debate: this.debate.getStats(),
      reflection: this.reflection.getStats(),
      skills: this.skillDiscovery.getStats(),
      curiosity: this.curiosity.getStats(),
      tools: this.tools.getStats(),
      goals: this.goals.getStats(),
      consciousness: this.consciousness.getSummary(),
    };
  }

  // ==================== HELPER METHODS ====================

  private assignSymbol(latent: number[]): string {
    const sum = latent.reduce((a, b) => a + b, 0);
    const normalized = Math.floor((sum + 10) * 0.5) % 12;
    return `S${normalized}`;
  }

  private extractAction(debateResult: DebateResult): number[] {
    const proposal = debateResult.winningProposal;

    if (proposal.content.steps && proposal.content.steps[0]?.action) {
      return proposal.content.steps[0].action;
    }

    if (proposal.content.alternatives && proposal.content.alternatives[0]?.action) {
      return proposal.content.alternatives[0].action;
    }

    return Array.from({ length: this.config.actionDim! }, () => Math.random() * 2 - 1);
  }

  private getHierarchyDepth(): number {
    const uniqueSymbols = new Set(this.previousSymbols).size;
    return Math.floor(Math.log2(uniqueSymbols + 1));
  }
}

// ==================== FACTORY ====================

export function createMultiDomainAgent(name: string, config?: Partial<AgentConfig>): MultiDomainAgent {
  return new MultiDomainAgent({
    name,
    ...config,
  });
}
