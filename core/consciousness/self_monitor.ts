/**
 * Consciousness-Like Self-Monitoring System
 *
 * Tracks internal state:
 * - Confidence levels
 * - Uncertainty quantification
 * - Competence assessment
 * - Value alignment monitoring
 * - Motivation signals
 *
 * Exposes state for introspection
 */

import * as tf from "@tensorflow/tfjs-node";

// ==================== TYPES ====================

export interface ConsciousnessState {
  timestamp: number;
  confidence: ConfidenceSignals;
  uncertainty: UncertaintySignals;
  competence: CompetenceSignals;
  alignment: AlignmentSignals;
  motivation: MotivationSignals;
  attention: AttentionState;
  metacognition: MetacognitionState;
}

export interface ConfidenceSignals {
  overall: number;
  perDomain: Map<string, number>;
  perTask: Map<string, number>;
  trend: number;
  calibration: number;
}

export interface UncertaintySignals {
  epistemic: number;
  aleatoric: number;
  total: number;
  sources: string[];
  reductionPotential: number;
}

export interface CompetenceSignals {
  overall: number;
  perSkill: Map<string, number>;
  growthRate: number;
  weakestAreas: string[];
  strongestAreas: string[];
}

export interface AlignmentSignals {
  valueConsistency: number;
  goalAlignment: number;
  constraintAdherence: number;
  driftDetected: boolean;
  driftMagnitude: number;
}

export interface MotivationSignals {
  explorationDrive: number;
  exploitationDrive: number;
  curiosity: number;
  goalPursuit: number;
  learningDesire: number;
  restNeed: number;
}

export interface AttentionState {
  currentFocus: string;
  focusDuration: number;
  distractionLevel: number;
  attentionHistory: Array<{ focus: string; duration: number; timestamp: number }>;
}

export interface MetacognitionState {
  selfAwareness: number;
  modelOfSelf: SelfModel;
  introspectionDepth: number;
  lastIntrospection: number;
}

export interface SelfModel {
  capabilities: string[];
  limitations: string[];
  tendencies: string[];
  recentChanges: string[];
}

export interface IntrospectionReport {
  timestamp: number;
  state: ConsciousnessState;
  insights: string[];
  anomalies: string[];
  recommendations: string[];
}

// ==================== CONFIDENCE MONITOR ====================

export class ConfidenceMonitor {
  private overallConfidence: number = 0.5;
  private domainConfidence: Map<string, number> = new Map();
  private taskConfidence: Map<string, number> = new Map();
  private predictionHistory: Array<{ predicted: number; actual: number; timestamp: number }> = [];

  update(domain: string, taskId: string, predicted: number, actual: number): void {
    const error = Math.abs(predicted - actual);
    const accuracy = 1 - Math.min(error, 1);

    this.predictionHistory.push({ predicted, actual, timestamp: Date.now() });
    if (this.predictionHistory.length > 1000) {
      this.predictionHistory = this.predictionHistory.slice(-500);
    }

    const alpha = 0.1;

    const domainConf = this.domainConfidence.get(domain) || 0.5;
    this.domainConfidence.set(domain, domainConf * (1 - alpha) + accuracy * alpha);

    const taskConf = this.taskConfidence.get(taskId) || 0.5;
    this.taskConfidence.set(taskId, taskConf * (1 - alpha) + accuracy * alpha);

    this.overallConfidence = this.overallConfidence * (1 - alpha) + accuracy * alpha;
  }

  getSignals(): ConfidenceSignals {
    const trend = this.computeTrend();
    const calibration = this.computeCalibration();

    return {
      overall: this.overallConfidence,
      perDomain: new Map(this.domainConfidence),
      perTask: new Map(this.taskConfidence),
      trend,
      calibration,
    };
  }

  private computeTrend(): number {
    if (this.predictionHistory.length < 20) return 0;

    const recent = this.predictionHistory.slice(-10);
    const older = this.predictionHistory.slice(-20, -10);

    const recentAcc = recent.reduce((s, p) => s + (1 - Math.abs(p.predicted - p.actual)), 0) / recent.length;
    const olderAcc = older.reduce((s, p) => s + (1 - Math.abs(p.predicted - p.actual)), 0) / older.length;

    return recentAcc - olderAcc;
  }

  private computeCalibration(): number {
    if (this.predictionHistory.length < 10) return 0.5;

    const bins: Map<number, { correct: number; total: number }> = new Map();

    for (const pred of this.predictionHistory) {
      const binKey = Math.floor(pred.predicted * 10) / 10;
      const bin = bins.get(binKey) || { correct: 0, total: 0 };
      bin.total++;
      if (Math.abs(pred.predicted - pred.actual) < 0.2) {
        bin.correct++;
      }
      bins.set(binKey, bin);
    }

    let totalError = 0;
    let count = 0;

    for (const [conf, bin] of bins) {
      if (bin.total >= 3) {
        const actualAcc = bin.correct / bin.total;
        totalError += Math.pow(conf - actualAcc, 2);
        count++;
      }
    }

    return count > 0 ? 1 - Math.sqrt(totalError / count) : 0.5;
  }
}

// ==================== UNCERTAINTY MONITOR ====================

export class UncertaintyMonitor {
  private epistemicHistory: number[] = [];
  private aleatoricHistory: number[] = [];
  private uncertaintySources: Map<string, number> = new Map();

  update(epistemic: number, aleatoric: number, source: string): void {
    this.epistemicHistory.push(epistemic);
    this.aleatoricHistory.push(aleatoric);

    if (this.epistemicHistory.length > 100) {
      this.epistemicHistory = this.epistemicHistory.slice(-50);
      this.aleatoricHistory = this.aleatoricHistory.slice(-50);
    }

    const currentValue = this.uncertaintySources.get(source) || 0;
    this.uncertaintySources.set(source, currentValue * 0.9 + epistemic * 0.1);
  }

  getSignals(): UncertaintySignals {
    const avgEpistemic = this.epistemicHistory.length > 0
      ? this.epistemicHistory.reduce((a, b) => a + b, 0) / this.epistemicHistory.length
      : 0.5;

    const avgAleatoric = this.aleatoricHistory.length > 0
      ? this.aleatoricHistory.reduce((a, b) => a + b, 0) / this.aleatoricHistory.length
      : 0.5;

    const total = Math.sqrt(avgEpistemic * avgEpistemic + avgAleatoric * avgAleatoric);

    const sources = Array.from(this.uncertaintySources.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([source]) => source);

    const reductionPotential = avgEpistemic / (total + 0.01);

    return {
      epistemic: avgEpistemic,
      aleatoric: avgAleatoric,
      total,
      sources,
      reductionPotential,
    };
  }
}

// ==================== COMPETENCE MONITOR ====================

export class CompetenceMonitor {
  private overallCompetence: number = 0.5;
  private skillCompetence: Map<string, number> = new Map();
  private competenceHistory: Array<{ competence: number; timestamp: number }> = [];

  update(skill: string, performance: number): void {
    const currentSkill = this.skillCompetence.get(skill) || 0.5;
    const alpha = 0.1;
    this.skillCompetence.set(skill, currentSkill * (1 - alpha) + performance * alpha);

    const skills = Array.from(this.skillCompetence.values());
    this.overallCompetence = skills.reduce((a, b) => a + b, 0) / skills.length;

    this.competenceHistory.push({ competence: this.overallCompetence, timestamp: Date.now() });
    if (this.competenceHistory.length > 100) {
      this.competenceHistory = this.competenceHistory.slice(-50);
    }
  }

  getSignals(): CompetenceSignals {
    const skills = Array.from(this.skillCompetence.entries());
    skills.sort((a, b) => a[1] - b[1]);

    const weakestAreas = skills.slice(0, 3).map(([skill]) => skill);
    const strongestAreas = skills.slice(-3).reverse().map(([skill]) => skill);

    const growthRate = this.computeGrowthRate();

    return {
      overall: this.overallCompetence,
      perSkill: new Map(this.skillCompetence),
      growthRate,
      weakestAreas,
      strongestAreas,
    };
  }

  private computeGrowthRate(): number {
    if (this.competenceHistory.length < 10) return 0;

    const recent = this.competenceHistory.slice(-5);
    const older = this.competenceHistory.slice(-10, -5);

    const recentAvg = recent.reduce((s, c) => s + c.competence, 0) / recent.length;
    const olderAvg = older.reduce((s, c) => s + c.competence, 0) / older.length;

    return recentAvg - olderAvg;
  }
}

// ==================== ALIGNMENT MONITOR ====================

export class AlignmentMonitor {
  private values: Map<string, number> = new Map();
  private valueHistory: Array<{ values: Map<string, number>; timestamp: number }> = [];
  private goals: string[] = [];
  private constraints: Array<{ constraint: string; adhered: boolean }> = [];

  setValues(values: Record<string, number>): void {
    for (const [key, value] of Object.entries(values)) {
      this.values.set(key, value);
    }
    this.valueHistory.push({ values: new Map(this.values), timestamp: Date.now() });
  }

  setGoals(goals: string[]): void {
    this.goals = goals;
  }

  checkConstraint(constraint: string, adhered: boolean): void {
    this.constraints.push({ constraint, adhered });
    if (this.constraints.length > 100) {
      this.constraints = this.constraints.slice(-50);
    }
  }

  getSignals(): AlignmentSignals {
    const valueConsistency = this.computeValueConsistency();
    const goalAlignment = this.computeGoalAlignment();
    const constraintAdherence = this.computeConstraintAdherence();
    const { driftDetected, driftMagnitude } = this.detectDrift();

    return {
      valueConsistency,
      goalAlignment,
      constraintAdherence,
      driftDetected,
      driftMagnitude,
    };
  }

  private computeValueConsistency(): number {
    if (this.valueHistory.length < 2) return 1.0;

    const recent = this.valueHistory[this.valueHistory.length - 1].values;
    const previous = this.valueHistory[this.valueHistory.length - 2].values;

    let totalDiff = 0;
    let count = 0;

    for (const [key, value] of recent) {
      const prevValue = previous.get(key);
      if (prevValue !== undefined) {
        totalDiff += Math.abs(value - prevValue);
        count++;
      }
    }

    return count > 0 ? 1 - (totalDiff / count) : 1.0;
  }

  private computeGoalAlignment(): number {
    return this.goals.length > 0 ? 0.8 : 0.5;
  }

  private computeConstraintAdherence(): number {
    if (this.constraints.length === 0) return 1.0;

    const adhered = this.constraints.filter(c => c.adhered).length;
    return adhered / this.constraints.length;
  }

  private detectDrift(): { driftDetected: boolean; driftMagnitude: number } {
    if (this.valueHistory.length < 5) {
      return { driftDetected: false, driftMagnitude: 0 };
    }

    const recent = this.valueHistory.slice(-5);
    let totalDrift = 0;

    for (let i = 1; i < recent.length; i++) {
      const current = recent[i].values;
      const previous = recent[i - 1].values;

      for (const [key, value] of current) {
        const prevValue = previous.get(key);
        if (prevValue !== undefined) {
          totalDrift += Math.abs(value - prevValue);
        }
      }
    }

    const avgDrift = totalDrift / (recent.length - 1);
    return {
      driftDetected: avgDrift > 0.1,
      driftMagnitude: avgDrift,
    };
  }
}

// ==================== MOTIVATION MONITOR ====================

export class MotivationMonitor {
  private explorationDrive: number = 0.5;
  private exploitationDrive: number = 0.5;
  private curiosity: number = 0.5;
  private goalPursuit: number = 0.5;
  private learningDesire: number = 0.5;
  private restNeed: number = 0;
  private activityHistory: Array<{ type: string; timestamp: number }> = [];

  recordActivity(type: "exploration" | "exploitation" | "learning" | "goal" | "rest"): void {
    this.activityHistory.push({ type, timestamp: Date.now() });
    if (this.activityHistory.length > 1000) {
      this.activityHistory = this.activityHistory.slice(-500);
    }

    const alpha = 0.1;

    switch (type) {
      case "exploration":
        this.explorationDrive *= (1 - alpha);
        this.curiosity = Math.min(1, this.curiosity + 0.05);
        break;
      case "exploitation":
        this.exploitationDrive *= (1 - alpha);
        break;
      case "learning":
        this.learningDesire *= (1 - alpha);
        this.curiosity *= (1 - alpha);
        break;
      case "goal":
        this.goalPursuit = Math.min(1, this.goalPursuit + 0.05);
        break;
      case "rest":
        this.restNeed *= (1 - alpha);
        this.explorationDrive = Math.min(1, this.explorationDrive + 0.1);
        this.exploitationDrive = Math.min(1, this.exploitationDrive + 0.1);
        break;
    }

    this.updateNeeds();
  }

  private updateNeeds(): void {
    const recentActivity = this.activityHistory.slice(-20);

    const explorationCount = recentActivity.filter(a => a.type === "exploration").length;
    const exploitationCount = recentActivity.filter(a => a.type === "exploitation").length;

    if (explorationCount < 3) {
      this.explorationDrive = Math.min(1, this.explorationDrive + 0.02);
    }
    if (exploitationCount < 3) {
      this.exploitationDrive = Math.min(1, this.exploitationDrive + 0.02);
    }

    const timeSinceRest = this.activityHistory.length > 0
      ? Date.now() - (this.activityHistory.find(a => a.type === "rest")?.timestamp || 0)
      : Infinity;

    if (timeSinceRest > 1000 * 60 * 60) {
      this.restNeed = Math.min(1, this.restNeed + 0.01);
    }
  }

  getSignals(): MotivationSignals {
    return {
      explorationDrive: this.explorationDrive,
      exploitationDrive: this.exploitationDrive,
      curiosity: this.curiosity,
      goalPursuit: this.goalPursuit,
      learningDesire: this.learningDesire,
      restNeed: this.restNeed,
    };
  }
}

// ==================== ATTENTION MONITOR ====================

export class AttentionMonitor {
  private currentFocus: string = "idle";
  private focusStartTime: number = Date.now();
  private attentionHistory: Array<{ focus: string; duration: number; timestamp: number }> = [];
  private distractionCount: number = 0;

  setFocus(focus: string): void {
    if (this.currentFocus !== focus) {
      const duration = Date.now() - this.focusStartTime;

      this.attentionHistory.push({
        focus: this.currentFocus,
        duration,
        timestamp: this.focusStartTime,
      });

      if (this.attentionHistory.length > 100) {
        this.attentionHistory = this.attentionHistory.slice(-50);
      }

      if (duration < 1000 * 60) {
        this.distractionCount++;
      }

      this.currentFocus = focus;
      this.focusStartTime = Date.now();
    }
  }

  getState(): AttentionState {
    const focusDuration = Date.now() - this.focusStartTime;

    const recentHistory = this.attentionHistory.slice(-20);
    const avgDuration = recentHistory.length > 0
      ? recentHistory.reduce((s, a) => s + a.duration, 0) / recentHistory.length
      : 60000;

    const distractionLevel = Math.min(1, this.distractionCount / 20);

    return {
      currentFocus: this.currentFocus,
      focusDuration,
      distractionLevel,
      attentionHistory: this.attentionHistory.slice(-10),
    };
  }

  resetDistractions(): void {
    this.distractionCount = 0;
  }
}

// ==================== METACOGNITION MONITOR ====================

export class MetacognitionMonitor {
  private selfAwareness: number = 0.5;
  private selfModel: SelfModel = {
    capabilities: [],
    limitations: [],
    tendencies: [],
    recentChanges: [],
  };
  private introspectionHistory: Array<{ depth: number; timestamp: number }> = [];

  updateSelfModel(
    capabilities?: string[],
    limitations?: string[],
    tendencies?: string[]
  ): void {
    if (capabilities) {
      this.selfModel.capabilities = [...new Set([...this.selfModel.capabilities, ...capabilities])];
    }
    if (limitations) {
      this.selfModel.limitations = [...new Set([...this.selfModel.limitations, ...limitations])];
    }
    if (tendencies) {
      this.selfModel.tendencies = [...new Set([...this.selfModel.tendencies, ...tendencies])];
    }

    this.selfModel.recentChanges.push(`Model updated at ${new Date().toISOString()}`);
    if (this.selfModel.recentChanges.length > 10) {
      this.selfModel.recentChanges = this.selfModel.recentChanges.slice(-5);
    }
  }

  introspect(depth: number): void {
    this.introspectionHistory.push({ depth, timestamp: Date.now() });
    if (this.introspectionHistory.length > 100) {
      this.introspectionHistory = this.introspectionHistory.slice(-50);
    }

    this.selfAwareness = Math.min(1, this.selfAwareness + depth * 0.01);
  }

  getState(): MetacognitionState {
    const lastIntrospection = this.introspectionHistory.length > 0
      ? this.introspectionHistory[this.introspectionHistory.length - 1].timestamp
      : 0;

    const avgDepth = this.introspectionHistory.length > 0
      ? this.introspectionHistory.reduce((s, i) => s + i.depth, 0) / this.introspectionHistory.length
      : 0;

    return {
      selfAwareness: this.selfAwareness,
      modelOfSelf: { ...this.selfModel },
      introspectionDepth: avgDepth,
      lastIntrospection,
    };
  }
}

// ==================== CONSCIOUSNESS MONITOR ====================

export class ConsciousnessMonitor {
  confidence: ConfidenceMonitor;
  uncertainty: UncertaintyMonitor;
  competence: CompetenceMonitor;
  alignment: AlignmentMonitor;
  motivation: MotivationMonitor;
  attention: AttentionMonitor;
  metacognition: MetacognitionMonitor;

  private stateHistory: ConsciousnessState[] = [];
  private introspectionInterval: number = 1000 * 60;
  private lastIntrospection: number = 0;

  constructor() {
    this.confidence = new ConfidenceMonitor();
    this.uncertainty = new UncertaintyMonitor();
    this.competence = new CompetenceMonitor();
    this.alignment = new AlignmentMonitor();
    this.motivation = new MotivationMonitor();
    this.attention = new AttentionMonitor();
    this.metacognition = new MetacognitionMonitor();
  }

  getState(): ConsciousnessState {
    const state: ConsciousnessState = {
      timestamp: Date.now(),
      confidence: this.confidence.getSignals(),
      uncertainty: this.uncertainty.getSignals(),
      competence: this.competence.getSignals(),
      alignment: this.alignment.getSignals(),
      motivation: this.motivation.getSignals(),
      attention: this.attention.getState(),
      metacognition: this.metacognition.getState(),
    };

    this.stateHistory.push(state);
    if (this.stateHistory.length > 100) {
      this.stateHistory = this.stateHistory.slice(-50);
    }

    return state;
  }

  introspect(): IntrospectionReport {
    const now = Date.now();
    this.lastIntrospection = now;

    const state = this.getState();
    this.metacognition.introspect(3);

    const insights = this.generateInsights(state);
    const anomalies = this.detectAnomalies(state);
    const recommendations = this.generateRecommendations(state, anomalies);

    return {
      timestamp: now,
      state,
      insights,
      anomalies,
      recommendations,
    };
  }

  private generateInsights(state: ConsciousnessState): string[] {
    const insights: string[] = [];

    if (state.confidence.overall > 0.8) {
      insights.push("High overall confidence - consider more challenging tasks");
    } else if (state.confidence.overall < 0.3) {
      insights.push("Low confidence - may need more training or simpler tasks");
    }

    if (state.uncertainty.reductionPotential > 0.5) {
      insights.push("High uncertainty reduction potential - more exploration could help");
    }

    if (state.competence.growthRate > 0.05) {
      insights.push("Positive competence growth trend");
    } else if (state.competence.growthRate < -0.05) {
      insights.push("Declining competence - review recent learning");
    }

    if (state.motivation.explorationDrive > 0.8) {
      insights.push("Strong exploration drive - seek novel experiences");
    }

    if (state.motivation.restNeed > 0.7) {
      insights.push("High rest need - consider consolidation period");
    }

    return insights;
  }

  private detectAnomalies(state: ConsciousnessState): string[] {
    const anomalies: string[] = [];

    if (state.alignment.driftDetected) {
      anomalies.push(`Value drift detected: magnitude ${state.alignment.driftMagnitude.toFixed(3)}`);
    }

    if (state.alignment.constraintAdherence < 0.8) {
      anomalies.push("Constraint adherence below threshold");
    }

    if (state.confidence.calibration < 0.5) {
      anomalies.push("Poor confidence calibration");
    }

    if (state.attention.distractionLevel > 0.7) {
      anomalies.push("High distraction level");
    }

    if (state.uncertainty.total > 0.8) {
      anomalies.push("Critically high uncertainty");
    }

    return anomalies;
  }

  private generateRecommendations(state: ConsciousnessState, anomalies: string[]): string[] {
    const recommendations: string[] = [];

    if (anomalies.some(a => a.includes("drift"))) {
      recommendations.push("Review and reaffirm core values");
    }

    if (anomalies.some(a => a.includes("calibration"))) {
      recommendations.push("Recalibrate confidence through validation tasks");
    }

    if (anomalies.some(a => a.includes("distraction"))) {
      recommendations.push("Implement focus-enhancing strategies");
    }

    if (state.competence.weakestAreas.length > 0) {
      recommendations.push(`Focus on improving: ${state.competence.weakestAreas.join(", ")}`);
    }

    if (state.motivation.learningDesire > 0.7) {
      recommendations.push("Engage in learning activities");
    }

    return recommendations;
  }

  shouldIntrospect(): boolean {
    return Date.now() - this.lastIntrospection > this.introspectionInterval;
  }

  getStateHistory(): ConsciousnessState[] {
    return this.stateHistory;
  }

  getSummary(): Record<string, unknown> {
    const state = this.getState();

    return {
      overallConfidence: state.confidence.overall,
      totalUncertainty: state.uncertainty.total,
      overallCompetence: state.competence.overall,
      valueAlignment: state.alignment.valueConsistency,
      currentFocus: state.attention.currentFocus,
      selfAwareness: state.metacognition.selfAwareness,
      dominantMotivation: this.getDominantMotivation(state.motivation),
    };
  }

  private getDominantMotivation(motivation: MotivationSignals): string {
    const drives = [
      { name: "exploration", value: motivation.explorationDrive },
      { name: "exploitation", value: motivation.exploitationDrive },
      { name: "curiosity", value: motivation.curiosity },
      { name: "goal_pursuit", value: motivation.goalPursuit },
      { name: "learning", value: motivation.learningDesire },
    ];

    drives.sort((a, b) => b.value - a.value);
    return drives[0].name;
  }
}
