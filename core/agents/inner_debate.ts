/**
 * Multi-Agent Inner Debate System
 *
 * Spawns internal agents for:
 * - Planner: Strategic action sequencing
 * - Critic: Identifies flaws and risks
 * - Verifier: Validates correctness
 * - Explainer: Generates reasoning chains
 * - Explorer: Proposes novel alternatives
 *
 * Decisions resolved via internal debate + voting
 */

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";

// ==================== TYPES ====================

export interface Proposal {
  id: string;
  agentId: string;
  content: any;
  confidence: number;
  reasoning: string[];
  timestamp: number;
}

export interface Critique {
  proposalId: string;
  criticId: string;
  issues: string[];
  severity: number;
  suggestions: string[];
}

export interface Vote {
  agentId: string;
  proposalId: string;
  score: number;
  reasoning: string;
}

export interface DebateResult {
  winningProposal: Proposal;
  votes: Vote[];
  consensus: number;
  debateRounds: number;
  dissent: string[];
}

export interface AgentState {
  id: string;
  role: AgentRole;
  confidence: number;
  expertise: Map<string, number>;
  recentDecisions: string[];
  successRate: number;
}

export type AgentRole = "planner" | "critic" | "verifier" | "explainer" | "explorer";

// ==================== BASE INNER AGENT ====================

abstract class InnerAgent {
  id: string;
  role: AgentRole;
  protected confidence: number = 0.5;
  protected model: tf.LayersModel;
  protected expertise: Map<string, number> = new Map();
  protected recentDecisions: Array<{ decision: any; outcome: number }> = [];

  constructor(role: AgentRole, inputDim: number, outputDim: number) {
    this.id = `${role}_${Math.random().toString(36).slice(2, 8)}`;
    this.role = role;
    this.model = this.buildModel(inputDim, outputDim);
  }

  protected buildModel(inputDim: number, outputDim: number): tf.LayersModel {
    const input = tf.input({ shape: [inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: outputDim, activation: "tanh" }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    return model;
  }

  abstract propose(context: number[], goal: any): Promise<Proposal>;
  abstract critique(proposal: Proposal): Promise<Critique>;
  abstract vote(proposals: Proposal[], context: number[]): Promise<Vote>;

  updateConfidence(outcome: number): void {
    this.confidence = this.confidence * 0.9 + outcome * 0.1;
  }

  recordDecision(decision: any, outcome: number): void {
    this.recentDecisions.push({ decision, outcome });
    if (this.recentDecisions.length > 100) {
      this.recentDecisions = this.recentDecisions.slice(-50);
    }
    this.updateConfidence(outcome);
  }

  getState(): AgentState {
    const successRate = this.recentDecisions.length > 0
      ? this.recentDecisions.reduce((s, d) => s + (d.outcome > 0.5 ? 1 : 0), 0) / this.recentDecisions.length
      : 0.5;

    return {
      id: this.id,
      role: this.role,
      confidence: this.confidence,
      expertise: this.expertise,
      recentDecisions: this.recentDecisions.slice(-10).map(d => JSON.stringify(d.decision)),
      successRate,
    };
  }
}

// ==================== PLANNER AGENT ====================

export class PlannerAgent extends InnerAgent {
  private planningHorizon: number = 10;

  constructor(inputDim = 64, outputDim = 32) {
    super("planner", inputDim, outputDim);
    this.expertise.set("sequencing", 0.7);
    this.expertise.set("goal_decomposition", 0.6);
  }

  async propose(context: number[], goal: any): Promise<Proposal> {
    const plan = await this.generatePlan(context, goal);

    return {
      id: `plan_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      agentId: this.id,
      content: plan,
      confidence: this.confidence,
      reasoning: this.generateReasoning(plan),
      timestamp: Date.now(),
    };
  }

  private async generatePlan(context: number[], goal: any): Promise<{
    steps: Array<{ action: number[]; expected: number[] }>;
    milestones: number[];
    contingencies: Array<{ trigger: string; action: string }>;
  }> {
    const steps: Array<{ action: number[]; expected: number[] }> = [];

    return tf.tidy(() => {
      const contextT = tf.tensor2d([this.padContext(context)]);
      const encoded = this.model.predict(contextT) as tf.Tensor;
      const encodedArr = Array.from(encoded.dataSync());

      for (let i = 0; i < this.planningHorizon; i++) {
        const stepInput = [...encodedArr.slice(0, 16), i / this.planningHorizon];
        const paddedInput = this.padToLength(stepInput, 64);

        const stepT = tf.tensor2d([paddedInput]);
        const stepOutput = this.model.predict(stepT) as tf.Tensor;
        const action = Array.from(stepOutput.dataSync()).slice(0, 4);

        steps.push({
          action,
          expected: action.map(a => a + Math.random() * 0.1),
        });
      }

      return {
        steps,
        milestones: [2, 5, 8],
        contingencies: [
          { trigger: "reward_below_threshold", action: "replan" },
          { trigger: "unexpected_state", action: "explore" },
        ],
      };
    });
  }

  private generateReasoning(plan: any): string[] {
    return [
      `Decomposed goal into ${plan.steps.length} sequential steps`,
      `Identified ${plan.milestones.length} critical milestones`,
      `Prepared ${plan.contingencies.length} contingency plans`,
      `Plan confidence: ${(this.confidence * 100).toFixed(1)}%`,
    ];
  }

  async critique(proposal: Proposal): Promise<Critique> {
    const issues: string[] = [];
    const suggestions: string[] = [];

    if (proposal.content.steps.length > 15) {
      issues.push("Plan may be overly complex");
      suggestions.push("Consider breaking into sub-goals");
    }

    if (proposal.confidence < 0.3) {
      issues.push("Low confidence in proposal");
      suggestions.push("Gather more information before execution");
    }

    return {
      proposalId: proposal.id,
      criticId: this.id,
      issues,
      severity: issues.length * 0.2,
      suggestions,
    };
  }

  async vote(proposals: Proposal[], context: number[]): Promise<Vote> {
    let bestProposal = proposals[0];
    let bestScore = -Infinity;

    for (const proposal of proposals) {
      const score = this.evaluateProposal(proposal, context);
      if (score > bestScore) {
        bestScore = score;
        bestProposal = proposal;
      }
    }

    return {
      agentId: this.id,
      proposalId: bestProposal.id,
      score: bestScore,
      reasoning: `Selected based on planning coherence and milestone clarity`,
    };
  }

  private evaluateProposal(proposal: Proposal, context: number[]): number {
    const hasSteps = proposal.content.steps?.length > 0 ? 0.3 : 0;
    const hasContingencies = proposal.content.contingencies?.length > 0 ? 0.2 : 0;
    const confidenceScore = proposal.confidence * 0.3;
    const coherenceScore = 0.2;

    return hasSteps + hasContingencies + confidenceScore + coherenceScore;
  }

  private padContext(context: number[]): number[] {
    return this.padToLength(context, 64);
  }

  private padToLength(arr: number[], length: number): number[] {
    if (arr.length >= length) return arr.slice(0, length);
    return [...arr, ...new Array(length - arr.length).fill(0)];
  }
}

// ==================== CRITIC AGENT ====================

export class CriticAgent extends InnerAgent {
  private criticalThreshold: number = 0.3;

  constructor(inputDim = 64, outputDim = 16) {
    super("critic", inputDim, outputDim);
    this.expertise.set("risk_assessment", 0.8);
    this.expertise.set("flaw_detection", 0.7);
  }

  async propose(context: number[], goal: any): Promise<Proposal> {
    const risks = this.identifyRisks(context);

    return {
      id: `critique_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      agentId: this.id,
      content: {
        risks,
        safetyConstraints: this.generateSafetyConstraints(risks),
        recommendations: this.generateRecommendations(risks),
      },
      confidence: this.confidence,
      reasoning: [`Identified ${risks.length} potential risks`, `Generated safety constraints`],
      timestamp: Date.now(),
    };
  }

  private identifyRisks(context: number[]): Array<{ type: string; severity: number; description: string }> {
    const risks: Array<{ type: string; severity: number; description: string }> = [];

    const variance = this.computeVariance(context);
    if (variance > 0.5) {
      risks.push({
        type: "high_variance",
        severity: variance,
        description: "High uncertainty in current state",
      });
    }

    const entropy = this.computeEntropy(context);
    if (entropy > 0.7) {
      risks.push({
        type: "high_entropy",
        severity: entropy,
        description: "State contains high information entropy",
      });
    }

    return risks;
  }

  private generateSafetyConstraints(risks: any[]): string[] {
    return risks.map(r => `Constraint: Mitigate ${r.type} risk`);
  }

  private generateRecommendations(risks: any[]): string[] {
    return risks.map(r => `Recommendation: Address ${r.description}`);
  }

  async critique(proposal: Proposal): Promise<Critique> {
    const issues: string[] = [];
    const suggestions: string[] = [];

    if (proposal.confidence < this.criticalThreshold) {
      issues.push(`Confidence ${(proposal.confidence * 100).toFixed(1)}% below threshold`);
      suggestions.push("Increase certainty before proceeding");
    }

    if (!proposal.content.steps && !proposal.content.risks) {
      issues.push("Proposal lacks actionable content");
      suggestions.push("Provide concrete steps or risk analysis");
    }

    if (proposal.reasoning.length < 2) {
      issues.push("Insufficient reasoning provided");
      suggestions.push("Elaborate on decision rationale");
    }

    return {
      proposalId: proposal.id,
      criticId: this.id,
      issues,
      severity: Math.min(issues.length * 0.25, 1.0),
      suggestions,
    };
  }

  async vote(proposals: Proposal[], context: number[]): Promise<Vote> {
    let bestProposal = proposals[0];
    let lowestRisk = Infinity;

    for (const proposal of proposals) {
      const riskScore = this.assessRisk(proposal);
      if (riskScore < lowestRisk) {
        lowestRisk = riskScore;
        bestProposal = proposal;
      }
    }

    return {
      agentId: this.id,
      proposalId: bestProposal.id,
      score: 1 - lowestRisk,
      reasoning: `Selected proposal with lowest risk score: ${lowestRisk.toFixed(3)}`,
    };
  }

  private assessRisk(proposal: Proposal): number {
    let risk = 0;

    risk += (1 - proposal.confidence) * 0.4;

    if (proposal.content.steps?.length > 10) {
      risk += 0.2;
    }

    if (!proposal.content.contingencies || proposal.content.contingencies.length === 0) {
      risk += 0.2;
    }

    return Math.min(risk, 1.0);
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
}

// ==================== VERIFIER AGENT ====================

export class VerifierAgent extends InnerAgent {
  private verificationThreshold: number = 0.7;

  constructor(inputDim = 64, outputDim = 8) {
    super("verifier", inputDim, outputDim);
    this.expertise.set("consistency_checking", 0.9);
    this.expertise.set("constraint_validation", 0.8);
  }

  async propose(context: number[], goal: any): Promise<Proposal> {
    const verificationResults = this.verify(context, goal);

    return {
      id: `verify_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      agentId: this.id,
      content: verificationResults,
      confidence: this.confidence,
      reasoning: [`Verification score: ${verificationResults.score.toFixed(3)}`, `Checked ${verificationResults.checksPerformed} constraints`],
      timestamp: Date.now(),
    };
  }

  private verify(context: number[], goal: any): {
    passed: boolean;
    score: number;
    checksPerformed: number;
    violations: string[];
  } {
    const violations: string[] = [];
    let checksPerformed = 0;
    let checksPassed = 0;

    checksPerformed++;
    if (context.length > 0) checksPassed++;
    else violations.push("Empty context");

    checksPerformed++;
    const hasNaN = context.some(x => isNaN(x));
    if (!hasNaN) checksPassed++;
    else violations.push("NaN values in context");

    checksPerformed++;
    const hasInf = context.some(x => !isFinite(x));
    if (!hasInf) checksPassed++;
    else violations.push("Infinite values in context");

    checksPerformed++;
    const bounded = context.every(x => Math.abs(x) < 1000);
    if (bounded) checksPassed++;
    else violations.push("Values exceed bounds");

    const score = checksPassed / checksPerformed;

    return {
      passed: score >= this.verificationThreshold,
      score,
      checksPerformed,
      violations,
    };
  }

  async critique(proposal: Proposal): Promise<Critique> {
    const issues: string[] = [];
    const suggestions: string[] = [];

    if (proposal.content.violations && proposal.content.violations.length > 0) {
      issues.push(...proposal.content.violations);
      suggestions.push("Address all constraint violations before proceeding");
    }

    if (proposal.content.score && proposal.content.score < this.verificationThreshold) {
      issues.push(`Verification score ${proposal.content.score.toFixed(3)} below threshold`);
      suggestions.push(`Improve to at least ${this.verificationThreshold}`);
    }

    return {
      proposalId: proposal.id,
      criticId: this.id,
      issues,
      severity: issues.length > 0 ? 0.8 : 0,
      suggestions,
    };
  }

  async vote(proposals: Proposal[], context: number[]): Promise<Vote> {
    let bestProposal = proposals[0];
    let highestVerification = -1;

    for (const proposal of proposals) {
      const verification = this.verifyProposal(proposal);
      if (verification > highestVerification) {
        highestVerification = verification;
        bestProposal = proposal;
      }
    }

    return {
      agentId: this.id,
      proposalId: bestProposal.id,
      score: highestVerification,
      reasoning: `Verified proposal with score: ${highestVerification.toFixed(3)}`,
    };
  }

  private verifyProposal(proposal: Proposal): number {
    let score = proposal.confidence * 0.4;

    if (proposal.reasoning.length >= 2) score += 0.2;
    if (proposal.content.checksPerformed && proposal.content.checksPerformed > 0) {
      score += (proposal.content.score || 0) * 0.4;
    } else {
      score += 0.2;
    }

    return Math.min(score, 1.0);
  }
}

// ==================== EXPLAINER AGENT ====================

export class ExplainerAgent extends InnerAgent {
  constructor(inputDim = 64, outputDim = 32) {
    super("explainer", inputDim, outputDim);
    this.expertise.set("reasoning_chains", 0.85);
    this.expertise.set("interpretation", 0.75);
  }

  async propose(context: number[], goal: any): Promise<Proposal> {
    const explanation = this.generateExplanation(context, goal);

    return {
      id: `explain_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      agentId: this.id,
      content: explanation,
      confidence: this.confidence,
      reasoning: explanation.reasoningChain,
      timestamp: Date.now(),
    };
  }

  private generateExplanation(context: number[], goal: any): {
    summary: string;
    reasoningChain: string[];
    keyFactors: Array<{ factor: string; importance: number }>;
    alternatives: string[];
  } {
    const importantIndices = this.findImportantFeatures(context);
    const keyFactors = importantIndices.map((idx, i) => ({
      factor: `feature_${idx}`,
      importance: context[idx] || 0,
    }));

    const reasoningChain = [
      `Analyzed context with ${context.length} dimensions`,
      `Identified ${keyFactors.length} key factors`,
      `Goal decomposition reveals ${typeof goal === "object" ? Object.keys(goal).length : 1} sub-objectives`,
      `Confidence in explanation: ${(this.confidence * 100).toFixed(1)}%`,
    ];

    return {
      summary: `Context analysis complete with ${keyFactors.length} key factors identified`,
      reasoningChain,
      keyFactors,
      alternatives: ["Consider alternative interpretation A", "Consider alternative interpretation B"],
    };
  }

  private findImportantFeatures(context: number[]): number[] {
    const indexed = context.map((v, i) => ({ v: Math.abs(v), i }));
    indexed.sort((a, b) => b.v - a.v);
    return indexed.slice(0, 5).map(x => x.i);
  }

  async critique(proposal: Proposal): Promise<Critique> {
    const issues: string[] = [];
    const suggestions: string[] = [];

    if (!proposal.content.reasoningChain || proposal.content.reasoningChain.length < 2) {
      issues.push("Insufficient reasoning chain");
      suggestions.push("Provide step-by-step reasoning");
    }

    if (!proposal.content.alternatives || proposal.content.alternatives.length === 0) {
      issues.push("No alternative explanations considered");
      suggestions.push("Consider at least 2 alternative interpretations");
    }

    return {
      proposalId: proposal.id,
      criticId: this.id,
      issues,
      severity: issues.length * 0.15,
      suggestions,
    };
  }

  async vote(proposals: Proposal[], context: number[]): Promise<Vote> {
    let bestProposal = proposals[0];
    let bestExplanation = -1;

    for (const proposal of proposals) {
      const explanationScore = this.scoreExplanation(proposal);
      if (explanationScore > bestExplanation) {
        bestExplanation = explanationScore;
        bestProposal = proposal;
      }
    }

    return {
      agentId: this.id,
      proposalId: bestProposal.id,
      score: bestExplanation,
      reasoning: `Selected most explanatory proposal with clarity score: ${bestExplanation.toFixed(3)}`,
    };
  }

  private scoreExplanation(proposal: Proposal): number {
    let score = 0;

    score += Math.min(proposal.reasoning.length * 0.1, 0.4);

    if (proposal.content.keyFactors) {
      score += Math.min(proposal.content.keyFactors.length * 0.05, 0.3);
    }

    if (proposal.content.alternatives) {
      score += Math.min(proposal.content.alternatives.length * 0.1, 0.2);
    }

    score += proposal.confidence * 0.1;

    return Math.min(score, 1.0);
  }
}

// ==================== EXPLORER AGENT ====================

export class ExplorerAgent extends InnerAgent {
  private explorationRate: number = 0.3;
  private noveltyThreshold: number = 0.5;

  constructor(inputDim = 64, outputDim = 32) {
    super("explorer", inputDim, outputDim);
    this.expertise.set("novelty_seeking", 0.9);
    this.expertise.set("alternative_generation", 0.85);
  }

  async propose(context: number[], goal: any): Promise<Proposal> {
    const alternatives = await this.generateAlternatives(context, goal);

    return {
      id: `explore_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      agentId: this.id,
      content: {
        alternatives,
        noveltyScores: alternatives.map(a => a.novelty),
        explorationPaths: this.generateExplorationPaths(context),
      },
      confidence: this.confidence,
      reasoning: [
        `Generated ${alternatives.length} novel alternatives`,
        `Average novelty score: ${(alternatives.reduce((s, a) => s + a.novelty, 0) / alternatives.length).toFixed(3)}`,
      ],
      timestamp: Date.now(),
    };
  }

  private async generateAlternatives(context: number[], goal: any): Promise<Array<{
    action: number[];
    novelty: number;
    expectedOutcome: string;
  }>> {
    const alternatives: Array<{ action: number[]; novelty: number; expectedOutcome: string }> = [];

    for (let i = 0; i < 5; i++) {
      const perturbedContext = context.map(v => v + (Math.random() - 0.5) * this.explorationRate);

      const action = perturbedContext.slice(0, 4).map(v => Math.tanh(v));
      const novelty = this.computeNovelty(action, alternatives.map(a => a.action));

      alternatives.push({
        action,
        novelty,
        expectedOutcome: `Alternative path ${i + 1}`,
      });
    }

    return alternatives.sort((a, b) => b.novelty - a.novelty);
  }

  private computeNovelty(action: number[], existingActions: number[][]): number {
    if (existingActions.length === 0) return 1.0;

    let minDistance = Infinity;
    for (const existing of existingActions) {
      let distance = 0;
      for (let i = 0; i < Math.min(action.length, existing.length); i++) {
        distance += Math.pow(action[i] - existing[i], 2);
      }
      minDistance = Math.min(minDistance, Math.sqrt(distance));
    }

    return Math.min(minDistance / 2, 1.0);
  }

  private generateExplorationPaths(context: number[]): Array<{ direction: string; potential: number }> {
    const paths: Array<{ direction: string; potential: number }> = [];

    for (let dim = 0; dim < Math.min(context.length, 4); dim++) {
      paths.push({
        direction: `increase_dim_${dim}`,
        potential: Math.random() * 0.5 + 0.3,
      });
      paths.push({
        direction: `decrease_dim_${dim}`,
        potential: Math.random() * 0.5 + 0.3,
      });
    }

    return paths.sort((a, b) => b.potential - a.potential).slice(0, 5);
  }

  async critique(proposal: Proposal): Promise<Critique> {
    const issues: string[] = [];
    const suggestions: string[] = [];

    const hasNovelty = proposal.content.noveltyScores?.some((n: number) => n > this.noveltyThreshold);
    if (!hasNovelty) {
      issues.push("Proposal lacks novelty");
      suggestions.push("Consider more diverse alternatives");
    }

    if (!proposal.content.explorationPaths || proposal.content.explorationPaths.length < 3) {
      issues.push("Insufficient exploration paths");
      suggestions.push("Generate more exploration directions");
    }

    return {
      proposalId: proposal.id,
      criticId: this.id,
      issues,
      severity: issues.length * 0.2,
      suggestions,
    };
  }

  async vote(proposals: Proposal[], context: number[]): Promise<Vote> {
    let bestProposal = proposals[0];
    let highestNovelty = -1;

    for (const proposal of proposals) {
      const noveltyScore = this.scoreNovelty(proposal);
      if (noveltyScore > highestNovelty) {
        highestNovelty = noveltyScore;
        bestProposal = proposal;
      }
    }

    return {
      agentId: this.id,
      proposalId: bestProposal.id,
      score: highestNovelty,
      reasoning: `Selected most novel proposal with novelty: ${highestNovelty.toFixed(3)}`,
    };
  }

  private scoreNovelty(proposal: Proposal): number {
    if (proposal.content.noveltyScores && proposal.content.noveltyScores.length > 0) {
      const avgNovelty = proposal.content.noveltyScores.reduce((s: number, n: number) => s + n, 0) /
        proposal.content.noveltyScores.length;
      return avgNovelty;
    }
    return proposal.confidence * 0.5;
  }
}

// ==================== DEBATE ORCHESTRATOR ====================

export class InnerDebateOrchestrator {
  private planner: PlannerAgent;
  private critic: CriticAgent;
  private verifier: VerifierAgent;
  private explainer: ExplainerAgent;
  private explorer: ExplorerAgent;

  private debateHistory: DebateResult[] = [];
  private maxDebateRounds: number = 5;
  private consensusThreshold: number = 0.7;

  constructor(inputDim = 64) {
    this.planner = new PlannerAgent(inputDim);
    this.critic = new CriticAgent(inputDim);
    this.verifier = new VerifierAgent(inputDim);
    this.explainer = new ExplainerAgent(inputDim);
    this.explorer = new ExplorerAgent(inputDim);
  }

  async debate(context: number[], goal: any): Promise<DebateResult> {
    const proposals: Proposal[] = [];
    const allCritiques: Critique[] = [];

    const [plannerProposal, criticProposal, verifierProposal, explainerProposal, explorerProposal] =
      await Promise.all([
        this.planner.propose(context, goal),
        this.critic.propose(context, goal),
        this.verifier.propose(context, goal),
        this.explainer.propose(context, goal),
        this.explorer.propose(context, goal),
      ]);

    proposals.push(plannerProposal, criticProposal, verifierProposal, explainerProposal, explorerProposal);

    let round = 0;
    let consensus = 0;

    while (round < this.maxDebateRounds && consensus < this.consensusThreshold) {
      const critiques = await Promise.all(
        proposals.map(p => Promise.all([
          this.planner.critique(p),
          this.critic.critique(p),
          this.verifier.critique(p),
          this.explainer.critique(p),
          this.explorer.critique(p),
        ]))
      );

      allCritiques.push(...critiques.flat());

      const votes = await Promise.all([
        this.planner.vote(proposals, context),
        this.critic.vote(proposals, context),
        this.verifier.vote(proposals, context),
        this.explainer.vote(proposals, context),
        this.explorer.vote(proposals, context),
      ]);

      const voteCount = new Map<string, { votes: number; totalScore: number }>();
      for (const vote of votes) {
        const existing = voteCount.get(vote.proposalId) || { votes: 0, totalScore: 0 };
        existing.votes++;
        existing.totalScore += vote.score;
        voteCount.set(vote.proposalId, existing);
      }

      let maxVotes = 0;
      let winningId = proposals[0].id;
      for (const [id, data] of voteCount) {
        if (data.votes > maxVotes || (data.votes === maxVotes && data.totalScore > (voteCount.get(winningId)?.totalScore || 0))) {
          maxVotes = data.votes;
          winningId = id;
        }
      }

      consensus = maxVotes / votes.length;
      round++;

      if (consensus >= this.consensusThreshold) {
        const winner = proposals.find(p => p.id === winningId)!;
        const dissent = votes.filter(v => v.proposalId !== winningId).map(v => v.reasoning);

        const result: DebateResult = {
          winningProposal: winner,
          votes,
          consensus,
          debateRounds: round,
          dissent,
        };

        this.debateHistory.push(result);
        return result;
      }
    }

    const result: DebateResult = {
      winningProposal: proposals[0],
      votes: [],
      consensus,
      debateRounds: round,
      dissent: ["No consensus reached"],
    };

    this.debateHistory.push(result);
    return result;
  }

  recordOutcome(proposalId: string, outcome: number): void {
    const agents = [this.planner, this.critic, this.verifier, this.explainer, this.explorer];

    for (const agent of agents) {
      const state = agent.getState();
      const wasContributor = state.recentDecisions.some(d => d.includes(proposalId));
      if (wasContributor) {
        agent.recordDecision(proposalId, outcome);
      }
    }
  }

  getAgentStates(): AgentState[] {
    return [
      this.planner.getState(),
      this.critic.getState(),
      this.verifier.getState(),
      this.explainer.getState(),
      this.explorer.getState(),
    ];
  }

  getDebateHistory(): DebateResult[] {
    return this.debateHistory;
  }

  getStats(): Record<string, unknown> {
    const states = this.getAgentStates();
    return {
      totalDebates: this.debateHistory.length,
      averageConsensus: this.debateHistory.reduce((s, d) => s + d.consensus, 0) / (this.debateHistory.length || 1),
      averageRounds: this.debateHistory.reduce((s, d) => s + d.debateRounds, 0) / (this.debateHistory.length || 1),
      agentConfidences: Object.fromEntries(states.map(s => [s.role, s.confidence])),
      agentSuccessRates: Object.fromEntries(states.map(s => [s.role, s.successRate])),
    };
  }
}

export { InnerAgent };
