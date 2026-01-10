/**
 * Long-Horizon Goal Tracker
 *
 * Supports:
 * - Persistent goals across days/weeks
 * - Goal decomposition into subgoals
 * - Sub-goal planning and tracking
 * - Progress estimation
 * - Strategy revision
 */

// ==================== TYPES ====================

export interface Goal {
  id: string;
  name: string;
  description: string;
  priority: number;
  status: GoalStatus;
  progress: number;
  subgoals: string[];
  parentGoal?: string;
  dependencies: string[];
  deadline?: number;
  createdAt: number;
  updatedAt: number;
  completedAt?: number;
  domain: string;
  estimatedEffort: number;
  actualEffort: number;
  strategy: GoalStrategy;
  milestones: Milestone[];
  metrics: GoalMetrics;
}

export type GoalStatus = "pending" | "active" | "blocked" | "completed" | "abandoned";

export interface GoalStrategy {
  approach: string;
  resources: string[];
  risks: string[];
  contingencies: string[];
  revisionCount: number;
}

export interface Milestone {
  id: string;
  name: string;
  targetProgress: number;
  achieved: boolean;
  achievedAt?: number;
}

export interface GoalMetrics {
  startProgress: number;
  progressHistory: Array<{ progress: number; timestamp: number }>;
  estimatedCompletion: number;
  confidenceInterval: { low: number; high: number };
  velocityTrend: number;
}

export interface GoalDecomposition {
  goalId: string;
  subgoals: Array<{
    name: string;
    description: string;
    estimatedEffort: number;
    dependencies: string[];
  }>;
  reasoning: string[];
}

export interface ProgressUpdate {
  goalId: string;
  oldProgress: number;
  newProgress: number;
  delta: number;
  timestamp: number;
  cause: string;
}

// ==================== GOAL REPOSITORY ====================

export class GoalRepository {
  private goals: Map<string, Goal> = new Map();
  private domainIndex: Map<string, string[]> = new Map();
  private statusIndex: Map<GoalStatus, string[]> = new Map();
  private hierarchyIndex: Map<string, string[]> = new Map();

  create(goal: Omit<Goal, "id" | "createdAt" | "updatedAt" | "progress" | "actualEffort" | "metrics">): string {
    const id = `goal_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    const fullGoal: Goal = {
      ...goal,
      id,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      progress: 0,
      actualEffort: 0,
      metrics: {
        startProgress: 0,
        progressHistory: [],
        estimatedCompletion: Date.now() + goal.estimatedEffort * 1000 * 60 * 60,
        confidenceInterval: { low: 0.5, high: 0.9 },
        velocityTrend: 0,
      },
    };

    this.goals.set(id, fullGoal);
    this.indexGoal(fullGoal);

    return id;
  }

  private indexGoal(goal: Goal): void {
    const domainGoals = this.domainIndex.get(goal.domain) || [];
    domainGoals.push(goal.id);
    this.domainIndex.set(goal.domain, domainGoals);

    const statusGoals = this.statusIndex.get(goal.status) || [];
    statusGoals.push(goal.id);
    this.statusIndex.set(goal.status, statusGoals);

    if (goal.parentGoal) {
      const children = this.hierarchyIndex.get(goal.parentGoal) || [];
      children.push(goal.id);
      this.hierarchyIndex.set(goal.parentGoal, children);
    }
  }

  get(id: string): Goal | undefined {
    return this.goals.get(id);
  }

  update(id: string, updates: Partial<Goal>): boolean {
    const goal = this.goals.get(id);
    if (!goal) return false;

    const oldStatus = goal.status;

    Object.assign(goal, updates, { updatedAt: Date.now() });

    if (updates.status && updates.status !== oldStatus) {
      const oldStatusGoals = this.statusIndex.get(oldStatus) || [];
      this.statusIndex.set(oldStatus, oldStatusGoals.filter(gid => gid !== id));

      const newStatusGoals = this.statusIndex.get(updates.status) || [];
      newStatusGoals.push(id);
      this.statusIndex.set(updates.status, newStatusGoals);
    }

    return true;
  }

  getByDomain(domain: string): Goal[] {
    const ids = this.domainIndex.get(domain) || [];
    return ids.map(id => this.goals.get(id)!).filter(Boolean);
  }

  getByStatus(status: GoalStatus): Goal[] {
    const ids = this.statusIndex.get(status) || [];
    return ids.map(id => this.goals.get(id)!).filter(Boolean);
  }

  getChildren(parentId: string): Goal[] {
    const childIds = this.hierarchyIndex.get(parentId) || [];
    return childIds.map(id => this.goals.get(id)!).filter(Boolean);
  }

  getActiveGoals(): Goal[] {
    return this.getByStatus("active");
  }

  getAll(): Goal[] {
    return Array.from(this.goals.values());
  }
}

// ==================== GOAL DECOMPOSER ====================

export class GoalDecomposer {
  decompose(goal: Goal): GoalDecomposition {
    const subgoals = this.generateSubgoals(goal);
    const reasoning = this.generateReasoning(goal, subgoals);

    return {
      goalId: goal.id,
      subgoals,
      reasoning,
    };
  }

  private generateSubgoals(goal: Goal): GoalDecomposition["subgoals"] {
    const complexity = this.estimateComplexity(goal);
    const numSubgoals = Math.max(2, Math.min(7, Math.ceil(complexity / 20)));

    const subgoals: GoalDecomposition["subgoals"] = [];

    subgoals.push({
      name: `${goal.name} - Research & Planning`,
      description: `Research requirements and plan approach for ${goal.name}`,
      estimatedEffort: goal.estimatedEffort * 0.15,
      dependencies: [],
    });

    for (let i = 1; i < numSubgoals - 1; i++) {
      subgoals.push({
        name: `${goal.name} - Phase ${i}`,
        description: `Implementation phase ${i} of ${numSubgoals - 2}`,
        estimatedEffort: goal.estimatedEffort * (0.7 / (numSubgoals - 2)),
        dependencies: [subgoals[i - 1]?.name || ""],
      });
    }

    subgoals.push({
      name: `${goal.name} - Validation & Completion`,
      description: `Validate and finalize ${goal.name}`,
      estimatedEffort: goal.estimatedEffort * 0.15,
      dependencies: [subgoals[subgoals.length - 1]?.name || ""],
    });

    return subgoals;
  }

  private estimateComplexity(goal: Goal): number {
    let complexity = 0;

    complexity += goal.estimatedEffort * 0.5;
    complexity += goal.dependencies.length * 10;
    complexity += goal.strategy.risks.length * 15;

    const descLength = goal.description.length;
    complexity += Math.log(descLength + 1) * 5;

    return complexity;
  }

  private generateReasoning(goal: Goal, subgoals: GoalDecomposition["subgoals"]): string[] {
    return [
      `Decomposed "${goal.name}" into ${subgoals.length} subgoals`,
      `Total estimated effort: ${goal.estimatedEffort} hours`,
      `Identified ${goal.dependencies.length} external dependencies`,
      `Strategy approach: ${goal.strategy.approach}`,
    ];
  }
}

// ==================== PROGRESS ESTIMATOR ====================

export class ProgressEstimator {
  private updateHistory: Map<string, ProgressUpdate[]> = new Map();

  estimate(goal: Goal): {
    estimatedCompletion: number;
    confidence: number;
    blockers: string[];
  } {
    const history = this.updateHistory.get(goal.id) || [];

    if (history.length < 2) {
      return {
        estimatedCompletion: goal.metrics.estimatedCompletion,
        confidence: 0.3,
        blockers: this.identifyBlockers(goal),
      };
    }

    const velocity = this.computeVelocity(history);
    const remainingProgress = 100 - goal.progress;

    if (velocity <= 0) {
      return {
        estimatedCompletion: Date.now() + 1000 * 60 * 60 * 24 * 30,
        confidence: 0.2,
        blockers: [...this.identifyBlockers(goal), "No positive progress velocity"],
      };
    }

    const hoursRemaining = remainingProgress / velocity;
    const estimatedCompletion = Date.now() + hoursRemaining * 1000 * 60 * 60;

    const confidence = this.computeConfidence(history, velocity);

    return {
      estimatedCompletion,
      confidence,
      blockers: this.identifyBlockers(goal),
    };
  }

  private computeVelocity(history: ProgressUpdate[]): number {
    if (history.length < 2) return 0;

    const recent = history.slice(-10);
    let totalDelta = 0;
    let totalTime = 0;

    for (let i = 1; i < recent.length; i++) {
      totalDelta += recent[i].delta;
      totalTime += (recent[i].timestamp - recent[i - 1].timestamp) / (1000 * 60 * 60);
    }

    return totalTime > 0 ? totalDelta / totalTime : 0;
  }

  private computeConfidence(history: ProgressUpdate[], velocity: number): number {
    if (history.length < 5) return 0.3;

    const deltas = history.map(h => h.delta);
    const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    const variance = deltas.reduce((s, d) => s + Math.pow(d - mean, 2), 0) / deltas.length;
    const cv = variance > 0 ? Math.sqrt(variance) / Math.abs(mean || 1) : 1;

    const consistency = 1 / (1 + cv);

    const dataDensity = Math.min(history.length / 20, 1);

    return consistency * 0.6 + dataDensity * 0.4;
  }

  private identifyBlockers(goal: Goal): string[] {
    const blockers: string[] = [];

    if (goal.status === "blocked") {
      blockers.push("Goal is explicitly blocked");
    }

    for (const depId of goal.dependencies) {
      blockers.push(`Depends on: ${depId}`);
    }

    if (goal.deadline && goal.deadline < Date.now()) {
      blockers.push("Past deadline");
    }

    return blockers;
  }

  recordUpdate(goalId: string, oldProgress: number, newProgress: number, cause: string): void {
    const update: ProgressUpdate = {
      goalId,
      oldProgress,
      newProgress,
      delta: newProgress - oldProgress,
      timestamp: Date.now(),
      cause,
    };

    const history = this.updateHistory.get(goalId) || [];
    history.push(update);
    this.updateHistory.set(goalId, history.slice(-100));
  }

  getUpdateHistory(goalId: string): ProgressUpdate[] {
    return this.updateHistory.get(goalId) || [];
  }
}

// ==================== STRATEGY REVISER ====================

export class StrategyReviser {
  revise(goal: Goal, reason: string): GoalStrategy {
    const currentStrategy = goal.strategy;

    const newApproach = this.generateNewApproach(currentStrategy, reason);
    const newResources = this.adjustResources(currentStrategy, reason);
    const newContingencies = this.generateContingencies(currentStrategy, reason);

    return {
      approach: newApproach,
      resources: newResources,
      risks: this.updateRisks(currentStrategy, reason),
      contingencies: newContingencies,
      revisionCount: currentStrategy.revisionCount + 1,
    };
  }

  private generateNewApproach(current: GoalStrategy, reason: string): string {
    const reasonLower = reason.toLowerCase();

    if (reasonLower.includes("slow") || reasonLower.includes("progress")) {
      return `${current.approach} (accelerated - focus on critical path)`;
    }

    if (reasonLower.includes("block") || reasonLower.includes("stuck")) {
      return `${current.approach} (pivoted - alternative approach)`;
    }

    if (reasonLower.includes("resource") || reasonLower.includes("constraint")) {
      return `${current.approach} (optimized - resource-efficient)`;
    }

    return `${current.approach} (revised v${current.revisionCount + 1})`;
  }

  private adjustResources(current: GoalStrategy, reason: string): string[] {
    const resources = [...current.resources];

    if (reason.includes("skill")) {
      resources.push("Additional training/learning resources");
    }

    if (reason.includes("time")) {
      resources.push("Time management tools");
    }

    return resources;
  }

  private updateRisks(current: GoalStrategy, reason: string): string[] {
    const risks = [...current.risks];

    if (!risks.some(r => r.includes(reason.slice(0, 20)))) {
      risks.push(`Risk identified: ${reason.slice(0, 50)}`);
    }

    return risks;
  }

  private generateContingencies(current: GoalStrategy, reason: string): string[] {
    const contingencies = [...current.contingencies];

    contingencies.push(`If ${reason}, then reassess approach`);

    return contingencies.slice(-5);
  }

  shouldRevise(goal: Goal): { shouldRevise: boolean; reasons: string[] } {
    const reasons: string[] = [];

    const history = goal.metrics.progressHistory;
    if (history.length >= 5) {
      const recentProgress = history.slice(-5);
      const avgDelta = recentProgress.reduce((s, h, i) =>
        i > 0 ? s + (h.progress - recentProgress[i - 1].progress) : s, 0
      ) / 4;

      if (avgDelta < 0.5) {
        reasons.push("Progress velocity too low");
      }
    }

    if (goal.status === "blocked") {
      reasons.push("Goal is blocked");
    }

    if (goal.deadline && goal.deadline < goal.metrics.estimatedCompletion) {
      reasons.push("Estimated completion exceeds deadline");
    }

    if (goal.actualEffort > goal.estimatedEffort * 1.5) {
      reasons.push("Actual effort exceeds estimate by 50%");
    }

    return {
      shouldRevise: reasons.length > 0,
      reasons,
    };
  }
}

// ==================== GOAL TRACKER ====================

export class GoalTracker {
  repository: GoalRepository;
  decomposer: GoalDecomposer;
  estimator: ProgressEstimator;
  reviser: StrategyReviser;

  private activeGoalId: string | null = null;
  private goalHistory: Array<{ goalId: string; action: string; timestamp: number }> = [];

  constructor() {
    this.repository = new GoalRepository();
    this.decomposer = new GoalDecomposer();
    this.estimator = new ProgressEstimator();
    this.reviser = new StrategyReviser();
  }

  createGoal(
    name: string,
    description: string,
    domain: string,
    estimatedEffort: number,
    options: {
      priority?: number;
      deadline?: number;
      dependencies?: string[];
      parentGoal?: string;
    } = {}
  ): string {
    const goalId = this.repository.create({
      name,
      description,
      priority: options.priority || 5,
      status: "pending",
      subgoals: [],
      parentGoal: options.parentGoal,
      dependencies: options.dependencies || [],
      deadline: options.deadline,
      domain,
      estimatedEffort,
      strategy: {
        approach: "Standard approach",
        resources: [],
        risks: [],
        contingencies: [],
        revisionCount: 0,
      },
      milestones: this.generateMilestones(name),
    });

    this.goalHistory.push({ goalId, action: "created", timestamp: Date.now() });

    return goalId;
  }

  private generateMilestones(name: string): Milestone[] {
    return [
      { id: `ms_25_${Date.now()}`, name: `${name} - 25% complete`, targetProgress: 25, achieved: false },
      { id: `ms_50_${Date.now()}`, name: `${name} - 50% complete`, targetProgress: 50, achieved: false },
      { id: `ms_75_${Date.now()}`, name: `${name} - 75% complete`, targetProgress: 75, achieved: false },
      { id: `ms_100_${Date.now()}`, name: `${name} - Complete`, targetProgress: 100, achieved: false },
    ];
  }

  activateGoal(goalId: string): boolean {
    const goal = this.repository.get(goalId);
    if (!goal) return false;

    if (this.activeGoalId) {
      this.repository.update(this.activeGoalId, { status: "pending" });
    }

    this.repository.update(goalId, { status: "active" });
    this.activeGoalId = goalId;

    this.goalHistory.push({ goalId, action: "activated", timestamp: Date.now() });

    return true;
  }

  updateProgress(goalId: string, progress: number, cause: string): boolean {
    const goal = this.repository.get(goalId);
    if (!goal) return false;

    const oldProgress = goal.progress;
    const newProgress = Math.max(0, Math.min(100, progress));

    this.estimator.recordUpdate(goalId, oldProgress, newProgress, cause);

    goal.metrics.progressHistory.push({ progress: newProgress, timestamp: Date.now() });

    for (const milestone of goal.milestones) {
      if (!milestone.achieved && newProgress >= milestone.targetProgress) {
        milestone.achieved = true;
        milestone.achievedAt = Date.now();
      }
    }

    const estimation = this.estimator.estimate(goal);

    this.repository.update(goalId, {
      progress: newProgress,
      metrics: {
        ...goal.metrics,
        estimatedCompletion: estimation.estimatedCompletion,
        confidenceInterval: {
          low: estimation.confidence * 0.8,
          high: Math.min(estimation.confidence * 1.2, 1),
        },
      },
    });

    if (newProgress >= 100) {
      this.completeGoal(goalId);
    }

    this.goalHistory.push({ goalId, action: `progress:${newProgress}`, timestamp: Date.now() });

    return true;
  }

  completeGoal(goalId: string): boolean {
    const goal = this.repository.get(goalId);
    if (!goal) return false;

    this.repository.update(goalId, {
      status: "completed",
      progress: 100,
      completedAt: Date.now(),
    });

    if (this.activeGoalId === goalId) {
      this.activeGoalId = null;
    }

    if (goal.parentGoal) {
      this.updateParentProgress(goal.parentGoal);
    }

    this.goalHistory.push({ goalId, action: "completed", timestamp: Date.now() });

    return true;
  }

  private updateParentProgress(parentId: string): void {
    const parent = this.repository.get(parentId);
    if (!parent) return;

    const children = this.repository.getChildren(parentId);
    if (children.length === 0) return;

    const avgProgress = children.reduce((s, c) => s + c.progress, 0) / children.length;
    this.updateProgress(parentId, avgProgress, "Subgoal progress aggregation");
  }

  decomposeGoal(goalId: string): string[] {
    const goal = this.repository.get(goalId);
    if (!goal) return [];

    const decomposition = this.decomposer.decompose(goal);
    const subgoalIds: string[] = [];

    for (const subgoal of decomposition.subgoals) {
      const subgoalId = this.createGoal(
        subgoal.name,
        subgoal.description,
        goal.domain,
        subgoal.estimatedEffort,
        {
          priority: goal.priority,
          parentGoal: goalId,
          dependencies: subgoal.dependencies,
        }
      );
      subgoalIds.push(subgoalId);
    }

    this.repository.update(goalId, { subgoals: subgoalIds });

    return subgoalIds;
  }

  reviseStrategy(goalId: string, reason: string): boolean {
    const goal = this.repository.get(goalId);
    if (!goal) return false;

    const newStrategy = this.reviser.revise(goal, reason);
    this.repository.update(goalId, { strategy: newStrategy });

    this.goalHistory.push({ goalId, action: `strategy_revised:${reason}`, timestamp: Date.now() });

    return true;
  }

  checkAndReviseStrategies(): Array<{ goalId: string; reasons: string[] }> {
    const revisions: Array<{ goalId: string; reasons: string[] }> = [];

    for (const goal of this.repository.getActiveGoals()) {
      const { shouldRevise, reasons } = this.reviser.shouldRevise(goal);

      if (shouldRevise) {
        const combinedReason = reasons.join("; ");
        this.reviseStrategy(goal.id, combinedReason);
        revisions.push({ goalId: goal.id, reasons });
      }
    }

    return revisions;
  }

  getActiveGoal(): Goal | null {
    return this.activeGoalId ? this.repository.get(this.activeGoalId) || null : null;
  }

  getGoalHierarchy(rootId: string): Goal & { children: any[] } {
    const goal = this.repository.get(rootId);
    if (!goal) return { ...this.createEmptyGoal(), children: [] };

    const children = this.repository.getChildren(rootId).map(c => this.getGoalHierarchy(c.id));

    return { ...goal, children };
  }

  private createEmptyGoal(): Goal {
    return {
      id: "",
      name: "",
      description: "",
      priority: 0,
      status: "pending",
      progress: 0,
      subgoals: [],
      dependencies: [],
      createdAt: 0,
      updatedAt: 0,
      domain: "",
      estimatedEffort: 0,
      actualEffort: 0,
      strategy: { approach: "", resources: [], risks: [], contingencies: [], revisionCount: 0 },
      milestones: [],
      metrics: {
        startProgress: 0,
        progressHistory: [],
        estimatedCompletion: 0,
        confidenceInterval: { low: 0, high: 0 },
        velocityTrend: 0,
      },
    };
  }

  getStats(): Record<string, unknown> {
    const goals = this.repository.getAll();
    const active = goals.filter(g => g.status === "active");
    const completed = goals.filter(g => g.status === "completed");

    return {
      totalGoals: goals.length,
      activeGoals: active.length,
      completedGoals: completed.length,
      avgProgress: goals.reduce((s, g) => s + g.progress, 0) / (goals.length || 1),
      avgCompletionTime: completed.length > 0
        ? completed.reduce((s, g) => s + ((g.completedAt || 0) - g.createdAt), 0) / completed.length
        : 0,
    };
  }
}
