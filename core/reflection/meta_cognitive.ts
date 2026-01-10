/**
 * Meta-Cognitive Self-Reflection Loop
 *
 * Enables:
 * - Self-critique of decisions and answers
 * - Failure pattern detection
 * - Strategy adjustment
 * - Architecture improvement proposals
 * - Per-domain competence tracking
 */

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";

// ==================== TYPES ====================

export interface Decision {
  id: string;
  timestamp: number;
  domain: string;
  context: number[];
  action: any;
  reasoning: string[];
  outcome?: DecisionOutcome;
}

export interface DecisionOutcome {
  success: boolean;
  reward: number;
  expectedReward: number;
  feedback?: string;
  timestamp: number;
}

export interface SelfCritique {
  decisionId: string;
  strengths: string[];
  weaknesses: string[];
  improvementSuggestions: string[];
  confidenceAdjustment: number;
  revisedStrategy?: string;
}

export interface FailurePattern {
  id: string;
  pattern: string;
  occurrences: number;
  domains: string[];
  triggerConditions: string[];
  suggestedFixes: string[];
  lastOccurrence: number;
}

export interface StrategyAdjustment {
  id: string;
  domain: string;
  previousStrategy: string;
  newStrategy: string;
  reason: string;
  expectedImprovement: number;
  appliedAt: number;
}

export interface ArchitectureProposal {
  id: string;
  component: string;
  currentState: string;
  proposedChange: string;
  rationale: string[];
  expectedBenefit: number;
  risk: number;
  implementationSteps: string[];
  status: "proposed" | "approved" | "rejected" | "implemented";
}

export interface DomainCompetence {
  domain: string;
  overallScore: number;
  subskills: Map<string, number>;
  recentPerformance: number[];
  improvementTrend: number;
  weakAreas: string[];
  strongAreas: string[];
  lastUpdated: number;
}

// ==================== DECISION TRACKER ====================

export class DecisionTracker {
  private decisions: Map<string, Decision> = new Map();
  private outcomes: Map<string, DecisionOutcome> = new Map();
  private domainDecisions: Map<string, string[]> = new Map();

  record(decision: Omit<Decision, "id" | "timestamp">): string {
    const id = `dec_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    const fullDecision: Decision = {
      ...decision,
      id,
      timestamp: Date.now(),
    };

    this.decisions.set(id, fullDecision);

    const domainDecs = this.domainDecisions.get(decision.domain) || [];
    domainDecs.push(id);
    this.domainDecisions.set(decision.domain, domainDecs);

    return id;
  }

  recordOutcome(decisionId: string, outcome: Omit<DecisionOutcome, "timestamp">): void {
    const fullOutcome: DecisionOutcome = {
      ...outcome,
      timestamp: Date.now(),
    };

    this.outcomes.set(decisionId, fullOutcome);

    const decision = this.decisions.get(decisionId);
    if (decision) {
      decision.outcome = fullOutcome;
    }
  }

  getDecision(id: string): Decision | undefined {
    return this.decisions.get(id);
  }

  getRecentDecisions(n = 10): Decision[] {
    const all = Array.from(this.decisions.values());
    all.sort((a, b) => b.timestamp - a.timestamp);
    return all.slice(0, n);
  }

  getDecisionsByDomain(domain: string, n = 10): Decision[] {
    const ids = this.domainDecisions.get(domain) || [];
    return ids
      .slice(-n)
      .map(id => this.decisions.get(id)!)
      .filter(Boolean)
      .reverse();
  }

  getFailedDecisions(n = 10): Decision[] {
    return Array.from(this.decisions.values())
      .filter(d => d.outcome && !d.outcome.success)
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, n);
  }

  getSuccessRate(domain?: string): number {
    let decisions = Array.from(this.decisions.values()).filter(d => d.outcome);

    if (domain) {
      decisions = decisions.filter(d => d.domain === domain);
    }

    if (decisions.length === 0) return 0.5;

    const successes = decisions.filter(d => d.outcome!.success).length;
    return successes / decisions.length;
  }

  getAverageReward(domain?: string): number {
    let decisions = Array.from(this.decisions.values()).filter(d => d.outcome);

    if (domain) {
      decisions = decisions.filter(d => d.domain === domain);
    }

    if (decisions.length === 0) return 0;

    return decisions.reduce((sum, d) => sum + d.outcome!.reward, 0) / decisions.length;
  }
}

// ==================== SELF CRITIC ====================

export class SelfCritic {
  private critiques: Map<string, SelfCritique> = new Map();
  private critiqueModel: tf.LayersModel;

  constructor(inputDim = 64) {
    this.critiqueModel = this.buildModel(inputDim);
  }

  private buildModel(inputDim: number): tf.LayersModel {
    const input = tf.input({ shape: [inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: 5, activation: "sigmoid" }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    return model;
  }

  async critique(decision: Decision): Promise<SelfCritique> {
    const strengths: string[] = [];
    const weaknesses: string[] = [];
    const suggestions: string[] = [];

    if (decision.reasoning.length >= 3) {
      strengths.push("Thorough reasoning provided");
    } else {
      weaknesses.push("Insufficient reasoning depth");
      suggestions.push("Provide more detailed reasoning chains");
    }

    if (decision.outcome) {
      if (decision.outcome.success) {
        strengths.push("Decision led to successful outcome");
      } else {
        weaknesses.push("Decision resulted in failure");
        suggestions.push("Analyze failure conditions and adjust approach");
      }

      const rewardGap = decision.outcome.expectedReward - decision.outcome.reward;
      if (Math.abs(rewardGap) > 0.3) {
        weaknesses.push(`Reward prediction off by ${rewardGap.toFixed(3)}`);
        suggestions.push("Improve reward prediction model");
      }
    }

    const contextAnalysis = await this.analyzeContext(decision.context);
    if (contextAnalysis.uncertaintyHigh) {
      weaknesses.push("Decision made under high uncertainty");
      suggestions.push("Gather more information before deciding");
    }

    if (contextAnalysis.novelSituation) {
      if (decision.outcome?.success) {
        strengths.push("Successfully handled novel situation");
      } else {
        weaknesses.push("Failed in novel situation");
        suggestions.push("Build more robust generalization");
      }
    }

    const confidenceAdjustment = this.computeConfidenceAdjustment(decision, strengths, weaknesses);

    const critique: SelfCritique = {
      decisionId: decision.id,
      strengths,
      weaknesses,
      improvementSuggestions: suggestions,
      confidenceAdjustment,
      revisedStrategy: weaknesses.length > 2 ? this.proposeRevisedStrategy(decision, weaknesses) : undefined,
    };

    this.critiques.set(decision.id, critique);
    return critique;
  }

  private async analyzeContext(context: number[]): Promise<{
    uncertaintyHigh: boolean;
    novelSituation: boolean;
  }> {
    const variance = this.computeVariance(context);
    const entropy = this.computeEntropy(context);

    return {
      uncertaintyHigh: variance > 0.5 || entropy > 0.7,
      novelSituation: entropy > 0.8,
    };
  }

  private computeConfidenceAdjustment(
    decision: Decision,
    strengths: string[],
    weaknesses: string[]
  ): number {
    let adjustment = 0;

    adjustment += strengths.length * 0.05;
    adjustment -= weaknesses.length * 0.08;

    if (decision.outcome) {
      adjustment += decision.outcome.success ? 0.1 : -0.15;
    }

    return Math.max(-0.5, Math.min(0.5, adjustment));
  }

  private proposeRevisedStrategy(decision: Decision, weaknesses: string[]): string {
    const strategies: string[] = [];

    if (weaknesses.some(w => w.includes("uncertainty"))) {
      strategies.push("increase_exploration");
    }
    if (weaknesses.some(w => w.includes("reasoning"))) {
      strategies.push("deeper_analysis");
    }
    if (weaknesses.some(w => w.includes("failure"))) {
      strategies.push("conservative_approach");
    }

    return strategies.join(", ") || "maintain_current";
  }

  private computeVariance(arr: number[]): number {
    if (arr.length === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  }

  private computeEntropy(arr: number[]): number {
    if (arr.length === 0) return 0;
    const sum = arr.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const probs = arr.map(x => Math.abs(x) / sum);
    return -probs.reduce((h, p) => h + (p > 0 ? p * Math.log2(p) : 0), 0) / Math.log2(arr.length || 2);
  }

  getCritiques(): SelfCritique[] {
    return Array.from(this.critiques.values());
  }
}

// ==================== FAILURE PATTERN DETECTOR ====================

export class FailurePatternDetector {
  private patterns: Map<string, FailurePattern> = new Map();
  private failureHistory: Array<{ decision: Decision; critique: SelfCritique }> = [];

  recordFailure(decision: Decision, critique: SelfCritique): void {
    this.failureHistory.push({ decision, critique });
    this.detectPatterns();
  }

  private detectPatterns(): void {
    if (this.failureHistory.length < 3) return;

    const recentFailures = this.failureHistory.slice(-20);

    this.detectWeaknessPattern(recentFailures);
    this.detectDomainPattern(recentFailures);
    this.detectContextPattern(recentFailures);
  }

  private detectWeaknessPattern(failures: Array<{ decision: Decision; critique: SelfCritique }>): void {
    const weaknessCount: Map<string, number> = new Map();

    for (const { critique } of failures) {
      for (const weakness of critique.weaknesses) {
        const key = this.normalizeWeakness(weakness);
        weaknessCount.set(key, (weaknessCount.get(key) || 0) + 1);
      }
    }

    for (const [weakness, count] of weaknessCount) {
      if (count >= 3) {
        this.addOrUpdatePattern({
          pattern: `recurring_weakness: ${weakness}`,
          occurrences: count,
          domains: [...new Set(failures.map(f => f.decision.domain))],
          triggerConditions: [`weakness: ${weakness}`],
          suggestedFixes: this.suggestFixForWeakness(weakness),
        });
      }
    }
  }

  private detectDomainPattern(failures: Array<{ decision: Decision; critique: SelfCritique }>): void {
    const domainFailures: Map<string, number> = new Map();

    for (const { decision } of failures) {
      domainFailures.set(decision.domain, (domainFailures.get(decision.domain) || 0) + 1);
    }

    for (const [domain, count] of domainFailures) {
      const totalInDomain = failures.filter(f => f.decision.domain === domain).length;
      if (count / totalInDomain > 0.7 && count >= 3) {
        this.addOrUpdatePattern({
          pattern: `domain_difficulty: ${domain}`,
          occurrences: count,
          domains: [domain],
          triggerConditions: [`domain: ${domain}`],
          suggestedFixes: [
            `Increase training data for ${domain}`,
            `Transfer knowledge from related domains`,
            `Reduce complexity of ${domain} tasks`,
          ],
        });
      }
    }
  }

  private detectContextPattern(failures: Array<{ decision: Decision; critique: SelfCritique }>): void {
    const highVarianceFailures = failures.filter(f => {
      const variance = this.computeVariance(f.decision.context);
      return variance > 0.6;
    });

    if (highVarianceFailures.length >= 3) {
      this.addOrUpdatePattern({
        pattern: "high_variance_context_failure",
        occurrences: highVarianceFailures.length,
        domains: [...new Set(highVarianceFailures.map(f => f.decision.domain))],
        triggerConditions: ["context_variance > 0.6"],
        suggestedFixes: [
          "Add context normalization",
          "Increase exploration in uncertain states",
          "Use ensemble methods for high variance",
        ],
      });
    }
  }

  private addOrUpdatePattern(data: Omit<FailurePattern, "id" | "lastOccurrence">): void {
    const existingId = Array.from(this.patterns.entries())
      .find(([_, p]) => p.pattern === data.pattern)?.[0];

    if (existingId) {
      const existing = this.patterns.get(existingId)!;
      existing.occurrences = data.occurrences;
      existing.domains = [...new Set([...existing.domains, ...data.domains])];
      existing.lastOccurrence = Date.now();
    } else {
      const id = `pattern_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      this.patterns.set(id, {
        ...data,
        id,
        lastOccurrence: Date.now(),
      });
    }
  }

  private normalizeWeakness(weakness: string): string {
    return weakness.toLowerCase().replace(/[^a-z_]/g, "_").slice(0, 50);
  }

  private suggestFixForWeakness(weakness: string): string[] {
    const fixes: string[] = [];

    if (weakness.includes("reasoning")) {
      fixes.push("Implement deeper reasoning chains");
      fixes.push("Add more intermediate validation steps");
    }
    if (weakness.includes("uncertainty")) {
      fixes.push("Improve uncertainty quantification");
      fixes.push("Add information-gathering actions");
    }
    if (weakness.includes("prediction")) {
      fixes.push("Retrain prediction models");
      fixes.push("Add ensemble predictions");
    }
    if (weakness.includes("failure")) {
      fixes.push("Analyze failure modes");
      fixes.push("Add failure recovery mechanisms");
    }

    return fixes.length > 0 ? fixes : ["General improvement needed"];
  }

  private computeVariance(arr: number[]): number {
    if (arr.length === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  }

  getPatterns(): FailurePattern[] {
    return Array.from(this.patterns.values());
  }

  getPatternsByDomain(domain: string): FailurePattern[] {
    return Array.from(this.patterns.values()).filter(p => p.domains.includes(domain));
  }

  getMostFrequentPattern(): FailurePattern | null {
    const patterns = Array.from(this.patterns.values());
    if (patterns.length === 0) return null;
    return patterns.sort((a, b) => b.occurrences - a.occurrences)[0];
  }
}

// ==================== STRATEGY ADJUSTER ====================

export class StrategyAdjuster {
  private strategies: Map<string, string> = new Map();
  private adjustments: StrategyAdjustment[] = [];
  private strategyPerformance: Map<string, { uses: number; successRate: number }> = new Map();

  adjust(
    domain: string,
    failurePatterns: FailurePattern[],
    competence: DomainCompetence
  ): StrategyAdjustment | null {
    const currentStrategy = this.strategies.get(domain) || "default";

    const shouldAdjust =
      competence.improvementTrend < 0 ||
      competence.overallScore < 0.4 ||
      failurePatterns.length > 2;

    if (!shouldAdjust) return null;

    const newStrategy = this.determineNewStrategy(domain, failurePatterns, competence, currentStrategy);

    if (newStrategy === currentStrategy) return null;

    const adjustment: StrategyAdjustment = {
      id: `adj_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      domain,
      previousStrategy: currentStrategy,
      newStrategy,
      reason: this.generateAdjustmentReason(failurePatterns, competence),
      expectedImprovement: this.estimateImprovement(newStrategy, competence),
      appliedAt: Date.now(),
    };

    this.strategies.set(domain, newStrategy);
    this.adjustments.push(adjustment);

    return adjustment;
  }

  private determineNewStrategy(
    domain: string,
    patterns: FailurePattern[],
    competence: DomainCompetence,
    current: string
  ): string {
    const strategyOptions = [
      "aggressive_exploration",
      "conservative_exploitation",
      "balanced",
      "transfer_focused",
      "ensemble_decision",
      "hierarchical_planning",
      "curiosity_driven",
    ];

    const hasHighVariancePattern = patterns.some(p => p.pattern.includes("variance"));
    const hasDomainDifficulty = patterns.some(p => p.pattern.includes("domain_difficulty"));
    const hasReasoningWeakness = patterns.some(p => p.pattern.includes("reasoning"));

    if (competence.overallScore < 0.3) {
      return "aggressive_exploration";
    }

    if (hasHighVariancePattern) {
      return "ensemble_decision";
    }

    if (hasDomainDifficulty) {
      return "transfer_focused";
    }

    if (hasReasoningWeakness) {
      return "hierarchical_planning";
    }

    if (competence.improvementTrend > 0.1) {
      return "conservative_exploitation";
    }

    return "balanced";
  }

  private generateAdjustmentReason(patterns: FailurePattern[], competence: DomainCompetence): string {
    const reasons: string[] = [];

    if (competence.overallScore < 0.4) {
      reasons.push(`Low competence (${(competence.overallScore * 100).toFixed(1)}%)`);
    }
    if (competence.improvementTrend < 0) {
      reasons.push(`Declining performance trend`);
    }
    if (patterns.length > 0) {
      reasons.push(`${patterns.length} failure patterns detected`);
    }

    return reasons.join("; ") || "Proactive optimization";
  }

  private estimateImprovement(newStrategy: string, competence: DomainCompetence): number {
    const strategyExpectedGains: Record<string, number> = {
      aggressive_exploration: 0.15,
      conservative_exploitation: 0.08,
      balanced: 0.1,
      transfer_focused: 0.12,
      ensemble_decision: 0.14,
      hierarchical_planning: 0.11,
      curiosity_driven: 0.13,
    };

    const baseGain = strategyExpectedGains[newStrategy] || 0.1;
    return baseGain * (1 - competence.overallScore);
  }

  recordStrategyPerformance(domain: string, success: boolean): void {
    const strategy = this.strategies.get(domain) || "default";
    const perf = this.strategyPerformance.get(strategy) || { uses: 0, successRate: 0.5 };

    perf.uses++;
    const alpha = 1 / perf.uses;
    perf.successRate = perf.successRate * (1 - alpha) + (success ? 1 : 0) * alpha;

    this.strategyPerformance.set(strategy, perf);
  }

  getStrategy(domain: string): string {
    return this.strategies.get(domain) || "default";
  }

  getAdjustments(): StrategyAdjustment[] {
    return this.adjustments;
  }

  getStrategyPerformance(): Map<string, { uses: number; successRate: number }> {
    return this.strategyPerformance;
  }
}

// ==================== ARCHITECTURE PROPOSER ====================

export class ArchitectureProposer {
  private proposals: ArchitectureProposal[] = [];
  private implementedChanges: string[] = [];

  propose(
    failurePatterns: FailurePattern[],
    competenceMap: Map<string, DomainCompetence>,
    systemStats: Record<string, unknown>
  ): ArchitectureProposal | null {
    const issues = this.identifyArchitecturalIssues(failurePatterns, competenceMap, systemStats);

    if (issues.length === 0) return null;

    const mostCritical = issues.sort((a, b) => b.severity - a.severity)[0];

    const proposal = this.generateProposal(mostCritical);
    this.proposals.push(proposal);

    return proposal;
  }

  private identifyArchitecturalIssues(
    patterns: FailurePattern[],
    competence: Map<string, DomainCompetence>,
    stats: Record<string, unknown>
  ): Array<{ component: string; issue: string; severity: number }> {
    const issues: Array<{ component: string; issue: string; severity: number }> = [];

    const avgCompetence = Array.from(competence.values())
      .reduce((s, c) => s + c.overallScore, 0) / (competence.size || 1);

    if (avgCompetence < 0.4) {
      issues.push({
        component: "learning_system",
        issue: "Overall learning efficiency too low",
        severity: 0.8,
      });
    }

    const highVariancePatterns = patterns.filter(p => p.pattern.includes("variance"));
    if (highVariancePatterns.length > 2) {
      issues.push({
        component: "world_model",
        issue: "High variance in predictions",
        severity: 0.7,
      });
    }

    const reasoningPatterns = patterns.filter(p => p.pattern.includes("reasoning"));
    if (reasoningPatterns.length > 1) {
      issues.push({
        component: "reasoning_engine",
        issue: "Insufficient reasoning depth",
        severity: 0.6,
      });
    }

    const domainDifficultyPatterns = patterns.filter(p => p.pattern.includes("domain"));
    if (domainDifficultyPatterns.length > 0) {
      issues.push({
        component: "transfer_system",
        issue: "Poor cross-domain transfer",
        severity: 0.65,
      });
    }

    return issues;
  }

  private generateProposal(issue: { component: string; issue: string; severity: number }): ArchitectureProposal {
    const proposals: Record<string, Partial<ArchitectureProposal>> = {
      learning_system: {
        proposedChange: "Add meta-learning layer with MAML-style adaptation",
        rationale: [
          "Current learning is too slow",
          "Meta-learning enables faster adaptation",
          "Few-shot learning capability added",
        ],
        implementationSteps: [
          "Create MetaLearner class",
          "Implement inner-loop and outer-loop optimization",
          "Integrate with existing learning pipeline",
          "Test on held-out domains",
        ],
        expectedBenefit: 0.25,
        risk: 0.3,
      },
      world_model: {
        proposedChange: "Add ensemble world models with uncertainty quantification",
        rationale: [
          "Single model has high variance",
          "Ensemble reduces prediction uncertainty",
          "Better calibrated confidence estimates",
        ],
        implementationSteps: [
          "Create EnsembleWorldModel class",
          "Train multiple models with different seeds",
          "Implement uncertainty aggregation",
          "Update planning to use ensemble",
        ],
        expectedBenefit: 0.2,
        risk: 0.25,
      },
      reasoning_engine: {
        proposedChange: "Add chain-of-thought reasoning with verification",
        rationale: [
          "Current reasoning is shallow",
          "CoT improves complex problem solving",
          "Verification catches errors early",
        ],
        implementationSteps: [
          "Create ChainOfThoughtReasoner class",
          "Implement step-by-step reasoning",
          "Add verification at each step",
          "Integrate with decision making",
        ],
        expectedBenefit: 0.22,
        risk: 0.2,
      },
      transfer_system: {
        proposedChange: "Add progressive transfer with domain similarity metrics",
        rationale: [
          "Current transfer is ad-hoc",
          "Similarity-based transfer more effective",
          "Progressive transfer reduces negative transfer",
        ],
        implementationSteps: [
          "Create DomainSimilarityMetric class",
          "Implement progressive knowledge transfer",
          "Add negative transfer detection",
          "Test cross-domain performance",
        ],
        expectedBenefit: 0.18,
        risk: 0.22,
      },
    };

    const template = proposals[issue.component] || {
      proposedChange: `Improve ${issue.component}`,
      rationale: [issue.issue],
      implementationSteps: ["Analyze issue", "Design solution", "Implement", "Test"],
      expectedBenefit: 0.15,
      risk: 0.25,
    };

    return {
      id: `arch_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      component: issue.component,
      currentState: issue.issue,
      proposedChange: template.proposedChange!,
      rationale: template.rationale!,
      expectedBenefit: template.expectedBenefit!,
      risk: template.risk!,
      implementationSteps: template.implementationSteps!,
      status: "proposed",
    };
  }

  approveProposal(proposalId: string): boolean {
    const proposal = this.proposals.find(p => p.id === proposalId);
    if (!proposal) return false;

    proposal.status = "approved";
    return true;
  }

  markImplemented(proposalId: string): boolean {
    const proposal = this.proposals.find(p => p.id === proposalId);
    if (!proposal || proposal.status !== "approved") return false;

    proposal.status = "implemented";
    this.implementedChanges.push(proposal.proposedChange);
    return true;
  }

  getProposals(): ArchitectureProposal[] {
    return this.proposals;
  }

  getPendingProposals(): ArchitectureProposal[] {
    return this.proposals.filter(p => p.status === "proposed" || p.status === "approved");
  }

  getImplementedChanges(): string[] {
    return this.implementedChanges;
  }
}

// ==================== COMPETENCE TRACKER ====================

export class CompetenceTracker {
  private competence: Map<string, DomainCompetence> = new Map();
  private performanceHistory: Map<string, number[]> = new Map();
  private windowSize: number = 50;

  update(domain: string, performance: number, subskill?: string): void {
    let comp = this.competence.get(domain);

    if (!comp) {
      comp = {
        domain,
        overallScore: 0.5,
        subskills: new Map(),
        recentPerformance: [],
        improvementTrend: 0,
        weakAreas: [],
        strongAreas: [],
        lastUpdated: Date.now(),
      };
      this.competence.set(domain, comp);
    }

    comp.recentPerformance.push(performance);
    if (comp.recentPerformance.length > this.windowSize) {
      comp.recentPerformance = comp.recentPerformance.slice(-this.windowSize);
    }

    const alpha = 0.1;
    comp.overallScore = comp.overallScore * (1 - alpha) + performance * alpha;

    if (subskill) {
      const currentSkill = comp.subskills.get(subskill) || 0.5;
      comp.subskills.set(subskill, currentSkill * (1 - alpha) + performance * alpha);
    }

    comp.improvementTrend = this.computeTrend(comp.recentPerformance);
    this.identifyStrengthsWeaknesses(comp);
    comp.lastUpdated = Date.now();

    const history = this.performanceHistory.get(domain) || [];
    history.push(performance);
    this.performanceHistory.set(domain, history.slice(-1000));
  }

  private computeTrend(performances: number[]): number {
    if (performances.length < 10) return 0;

    const recent = performances.slice(-10);
    const older = performances.slice(-20, -10);

    if (older.length === 0) return 0;

    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;

    return recentAvg - olderAvg;
  }

  private identifyStrengthsWeaknesses(comp: DomainCompetence): void {
    const subskillArray = Array.from(comp.subskills.entries());

    comp.strongAreas = subskillArray
      .filter(([_, score]) => score > 0.7)
      .map(([name]) => name);

    comp.weakAreas = subskillArray
      .filter(([_, score]) => score < 0.4)
      .map(([name]) => name);
  }

  getCompetence(domain: string): DomainCompetence | undefined {
    return this.competence.get(domain);
  }

  getAllCompetences(): DomainCompetence[] {
    return Array.from(this.competence.values());
  }

  getCompetenceMap(): Map<string, DomainCompetence> {
    return this.competence;
  }

  getOverallCompetence(): number {
    const comps = Array.from(this.competence.values());
    if (comps.length === 0) return 0.5;
    return comps.reduce((s, c) => s + c.overallScore, 0) / comps.length;
  }

  getWeakestDomains(n = 3): DomainCompetence[] {
    return Array.from(this.competence.values())
      .sort((a, b) => a.overallScore - b.overallScore)
      .slice(0, n);
  }

  getStrongestDomains(n = 3): DomainCompetence[] {
    return Array.from(this.competence.values())
      .sort((a, b) => b.overallScore - a.overallScore)
      .slice(0, n);
  }
}

// ==================== META COGNITIVE LOOP ====================

export class MetaCognitiveLoop {
  decisionTracker: DecisionTracker;
  selfCritic: SelfCritic;
  failureDetector: FailurePatternDetector;
  strategyAdjuster: StrategyAdjuster;
  architectureProposer: ArchitectureProposer;
  competenceTracker: CompetenceTracker;

  private reflectionInterval: number = 1000 * 60;
  private lastReflection: number = 0;

  constructor(inputDim = 64) {
    this.decisionTracker = new DecisionTracker();
    this.selfCritic = new SelfCritic(inputDim);
    this.failureDetector = new FailurePatternDetector();
    this.strategyAdjuster = new StrategyAdjuster();
    this.architectureProposer = new ArchitectureProposer();
    this.competenceTracker = new CompetenceTracker();
  }

  async recordDecision(
    domain: string,
    context: number[],
    action: any,
    reasoning: string[]
  ): Promise<string> {
    return this.decisionTracker.record({ domain, context, action, reasoning });
  }

  async recordOutcome(
    decisionId: string,
    success: boolean,
    reward: number,
    expectedReward: number,
    feedback?: string
  ): Promise<SelfCritique> {
    this.decisionTracker.recordOutcome(decisionId, {
      success,
      reward,
      expectedReward,
      feedback,
    });

    const decision = this.decisionTracker.getDecision(decisionId);
    if (!decision) throw new Error(`Decision ${decisionId} not found`);

    this.competenceTracker.update(decision.domain, success ? reward : 0);

    const critique = await this.selfCritic.critique(decision);

    if (!success) {
      this.failureDetector.recordFailure(decision, critique);
    }

    this.strategyAdjuster.recordStrategyPerformance(decision.domain, success);

    return critique;
  }

  async reflect(): Promise<{
    patterns: FailurePattern[];
    adjustments: StrategyAdjustment[];
    proposals: ArchitectureProposal | null;
    competence: DomainCompetence[];
  }> {
    const now = Date.now();
    if (now - this.lastReflection < this.reflectionInterval) {
      return {
        patterns: [],
        adjustments: [],
        proposals: null,
        competence: this.competenceTracker.getAllCompetences(),
      };
    }

    this.lastReflection = now;

    const patterns = this.failureDetector.getPatterns();
    const adjustments: StrategyAdjustment[] = [];

    for (const comp of this.competenceTracker.getAllCompetences()) {
      const domainPatterns = this.failureDetector.getPatternsByDomain(comp.domain);
      const adjustment = this.strategyAdjuster.adjust(comp.domain, domainPatterns, comp);
      if (adjustment) {
        adjustments.push(adjustment);
      }
    }

    const proposal = this.architectureProposer.propose(
      patterns,
      this.competenceTracker.getCompetenceMap(),
      this.getStats()
    );

    return {
      patterns,
      adjustments,
      proposals: proposal,
      competence: this.competenceTracker.getAllCompetences(),
    };
  }

  getStrategy(domain: string): string {
    return this.strategyAdjuster.getStrategy(domain);
  }

  getStats(): Record<string, unknown> {
    return {
      totalDecisions: this.decisionTracker.getRecentDecisions(1000).length,
      overallSuccessRate: this.decisionTracker.getSuccessRate(),
      overallCompetence: this.competenceTracker.getOverallCompetence(),
      failurePatterns: this.failureDetector.getPatterns().length,
      strategyAdjustments: this.strategyAdjuster.getAdjustments().length,
      architectureProposals: this.architectureProposer.getProposals().length,
      weakestDomains: this.competenceTracker.getWeakestDomains(3).map(d => d.domain),
      strongestDomains: this.competenceTracker.getStrongestDomains(3).map(d => d.domain),
    };
  }
}
