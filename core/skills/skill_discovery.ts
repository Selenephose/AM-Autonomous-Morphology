/**
 * Autonomous Skill Discovery System
 *
 * Enables:
 * - Clustering repeated behaviors
 * - Compressing into reusable programs
 * - Assigning names and semantics
 * - Reusing skills as tools in planning
 */

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";

// ==================== TYPES ====================

export interface Behavior {
  id: string;
  sequence: BehaviorStep[];
  context: number[];
  outcome: BehaviorOutcome;
  domain: string;
  timestamp: number;
}

export interface BehaviorStep {
  state: number[];
  action: number[];
  latent: number[];
  symbol: string;
  reward: number;
}

export interface BehaviorOutcome {
  totalReward: number;
  success: boolean;
  goalReached: boolean;
  duration: number;
}

export interface DiscoveredSkill {
  id: string;
  name: string;
  description: string;
  symbolPattern: string[];
  actionPrototype: number[][];
  preconditions: SkillPrecondition;
  effects: SkillEffects;
  successRate: number;
  usageCount: number;
  domains: string[];
  composability: number;
  abstractionLevel: number;
  createdAt: number;
  lastUsed: number;
}

export interface SkillPrecondition {
  stateRange: { min: number[]; max: number[] };
  requiredSymbols: string[];
  minConfidence: number;
}

export interface SkillEffects {
  expectedStateChange: number[];
  expectedReward: number;
  expectedDuration: number;
  sideEffects: string[];
}

export interface SkillCluster {
  id: string;
  centroid: number[];
  members: string[];
  cohesion: number;
  representativeSkill?: string;
}

// ==================== BEHAVIOR BUFFER ====================

export class BehaviorBuffer {
  private behaviors: Map<string, Behavior> = new Map();
  private domainIndex: Map<string, string[]> = new Map();
  private maxSize: number;

  constructor(maxSize = 10000) {
    this.maxSize = maxSize;
  }

  add(behavior: Omit<Behavior, "id" | "timestamp">): string {
    const id = `beh_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    const fullBehavior: Behavior = {
      ...behavior,
      id,
      timestamp: Date.now(),
    };

    this.behaviors.set(id, fullBehavior);

    const domainBehaviors = this.domainIndex.get(behavior.domain) || [];
    domainBehaviors.push(id);
    this.domainIndex.set(behavior.domain, domainBehaviors);

    if (this.behaviors.size > this.maxSize) {
      this.prune();
    }

    return id;
  }

  private prune(): void {
    const behaviors = Array.from(this.behaviors.entries());
    behaviors.sort((a, b) => {
      const scoreA = a[1].outcome.success ? 1 : 0 + a[1].outcome.totalReward;
      const scoreB = b[1].outcome.success ? 1 : 0 + b[1].outcome.totalReward;
      return scoreB - scoreA;
    });

    const toKeep = behaviors.slice(0, Math.floor(this.maxSize * 0.8));
    this.behaviors = new Map(toKeep);

    for (const [domain, ids] of this.domainIndex) {
      const kept = ids.filter(id => this.behaviors.has(id));
      this.domainIndex.set(domain, kept);
    }
  }

  getSuccessfulBehaviors(minReward = 0): Behavior[] {
    return Array.from(this.behaviors.values())
      .filter(b => b.outcome.success && b.outcome.totalReward >= minReward);
  }

  getByDomain(domain: string): Behavior[] {
    const ids = this.domainIndex.get(domain) || [];
    return ids.map(id => this.behaviors.get(id)!).filter(Boolean);
  }

  getBehavior(id: string): Behavior | undefined {
    return this.behaviors.get(id);
  }

  size(): number {
    return this.behaviors.size;
  }
}

// ==================== BEHAVIOR ENCODER ====================

export class BehaviorEncoder {
  private encoder: tf.LayersModel;
  private latentDim: number;

  constructor(inputDim = 64, latentDim = 32) {
    this.latentDim = latentDim;
    this.encoder = this.buildEncoder(inputDim, latentDim);
  }

  private buildEncoder(inputDim: number, latentDim: number): tf.LayersModel {
    const input = tf.input({ shape: [inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1) as tf.SymbolicTensor;
    const latent = tf.layers.dense({ units: latentDim }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: latent });
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    return model;
  }

  encode(behavior: Behavior): number[] {
    const features = this.extractFeatures(behavior);

    return tf.tidy(() => {
      const inputT = tf.tensor2d([this.padToLength(features, 64)]);
      const latentT = this.encoder.predict(inputT) as tf.Tensor;
      return Array.from(latentT.dataSync());
    });
  }

  private extractFeatures(behavior: Behavior): number[] {
    const features: number[] = [];

    features.push(behavior.sequence.length / 100);
    features.push(behavior.outcome.totalReward);
    features.push(behavior.outcome.success ? 1 : 0);
    features.push(behavior.outcome.duration / 1000);

    const avgReward = behavior.sequence.reduce((s, step) => s + step.reward, 0) / (behavior.sequence.length || 1);
    features.push(avgReward);

    if (behavior.sequence.length > 0) {
      const firstState = behavior.sequence[0].state;
      const lastState = behavior.sequence[behavior.sequence.length - 1].state;
      features.push(...firstState.slice(0, 8));
      features.push(...lastState.slice(0, 8));
    }

    const symbols = behavior.sequence.map(s => s.symbol);
    const uniqueSymbols = new Set(symbols).size;
    features.push(uniqueSymbols / 20);

    const transitions = new Set(symbols.map((s, i) => i > 0 ? `${symbols[i-1]}-${s}` : "")).size;
    features.push(transitions / 50);

    return features;
  }

  private padToLength(arr: number[], length: number): number[] {
    if (arr.length >= length) return arr.slice(0, length);
    return [...arr, ...new Array(length - arr.length).fill(0)];
  }

  getLatentDim(): number {
    return this.latentDim;
  }
}

// ==================== SKILL CLUSTERER ====================

export class SkillClusterer {
  private clusters: Map<string, SkillCluster> = new Map();
  private behaviorToCluster: Map<string, string> = new Map();
  private minClusterSize: number;
  private similarityThreshold: number;

  constructor(minClusterSize = 3, similarityThreshold = 0.7) {
    this.minClusterSize = minClusterSize;
    this.similarityThreshold = similarityThreshold;
  }

  cluster(encodedBehaviors: Array<{ id: string; encoding: number[] }>): SkillCluster[] {
    this.clusters.clear();
    this.behaviorToCluster.clear();

    for (const { id, encoding } of encodedBehaviors) {
      const nearestCluster = this.findNearestCluster(encoding);

      if (nearestCluster && this.similarity(encoding, nearestCluster.centroid) >= this.similarityThreshold) {
        this.addToCluster(nearestCluster.id, id, encoding);
      } else {
        this.createCluster(id, encoding);
      }
    }

    return this.getValidClusters();
  }

  private findNearestCluster(encoding: number[]): SkillCluster | null {
    let nearest: SkillCluster | null = null;
    let maxSimilarity = -1;

    for (const cluster of this.clusters.values()) {
      const sim = this.similarity(encoding, cluster.centroid);
      if (sim > maxSimilarity) {
        maxSimilarity = sim;
        nearest = cluster;
      }
    }

    return nearest;
  }

  private createCluster(behaviorId: string, encoding: number[]): void {
    const clusterId = `cluster_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    this.clusters.set(clusterId, {
      id: clusterId,
      centroid: [...encoding],
      members: [behaviorId],
      cohesion: 1.0,
    });

    this.behaviorToCluster.set(behaviorId, clusterId);
  }

  private addToCluster(clusterId: string, behaviorId: string, encoding: number[]): void {
    const cluster = this.clusters.get(clusterId);
    if (!cluster) return;

    cluster.members.push(behaviorId);
    this.behaviorToCluster.set(behaviorId, clusterId);

    const n = cluster.members.length;
    cluster.centroid = cluster.centroid.map((c, i) =>
      c * (n - 1) / n + encoding[i] / n
    );

    cluster.cohesion = this.computeCohesion(cluster);
  }

  private computeCohesion(cluster: SkillCluster): number {
    return Math.min(1.0, cluster.members.length / 10);
  }

  private similarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  private getValidClusters(): SkillCluster[] {
    return Array.from(this.clusters.values())
      .filter(c => c.members.length >= this.minClusterSize)
      .sort((a, b) => b.cohesion - a.cohesion);
  }

  getClusterForBehavior(behaviorId: string): string | undefined {
    return this.behaviorToCluster.get(behaviorId);
  }
}

// ==================== SKILL SYNTHESIZER ====================

export class SkillSynthesizer {
  private nameGenerator: SkillNameGenerator;
  private skillCounter: number = 0;

  constructor() {
    this.nameGenerator = new SkillNameGenerator();
  }

  synthesize(
    cluster: SkillCluster,
    behaviors: Behavior[],
    domain: string
  ): DiscoveredSkill {
    const symbolPattern = this.extractSymbolPattern(behaviors);
    const actionPrototype = this.extractActionPrototype(behaviors);
    const preconditions = this.inferPreconditions(behaviors);
    const effects = this.inferEffects(behaviors);

    const successfulBehaviors = behaviors.filter(b => b.outcome.success);
    const successRate = successfulBehaviors.length / behaviors.length;

    const name = this.nameGenerator.generate(symbolPattern, effects, domain);
    const description = this.generateDescription(symbolPattern, effects, successRate);

    this.skillCounter++;

    return {
      id: `skill_${this.skillCounter}_${Math.random().toString(36).slice(2, 6)}`,
      name,
      description,
      symbolPattern,
      actionPrototype,
      preconditions,
      effects,
      successRate,
      usageCount: behaviors.length,
      domains: [domain],
      composability: this.computeComposability(symbolPattern),
      abstractionLevel: this.computeAbstractionLevel(behaviors),
      createdAt: Date.now(),
      lastUsed: Date.now(),
    };
  }

  private extractSymbolPattern(behaviors: Behavior[]): string[] {
    const patternCounts: Map<string, number> = new Map();

    for (const behavior of behaviors) {
      const pattern = behavior.sequence.map(s => s.symbol).join(",");
      patternCounts.set(pattern, (patternCounts.get(pattern) || 0) + 1);
    }

    const sortedPatterns = Array.from(patternCounts.entries())
      .sort((a, b) => b[1] - a[1]);

    if (sortedPatterns.length > 0) {
      return sortedPatterns[0][0].split(",");
    }

    const allSymbols = behaviors.flatMap(b => b.sequence.map(s => s.symbol));
    const symbolFreq: Map<string, number> = new Map();
    for (const sym of allSymbols) {
      symbolFreq.set(sym, (symbolFreq.get(sym) || 0) + 1);
    }

    return Array.from(symbolFreq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([sym]) => sym);
  }

  private extractActionPrototype(behaviors: Behavior[]): number[][] {
    const avgLength = Math.round(
      behaviors.reduce((s, b) => s + b.sequence.length, 0) / behaviors.length
    );

    const prototype: number[][] = [];

    for (let step = 0; step < avgLength; step++) {
      const actionsAtStep = behaviors
        .filter(b => b.sequence[step])
        .map(b => b.sequence[step].action);

      if (actionsAtStep.length > 0) {
        const avgAction = this.averageVectors(actionsAtStep);
        prototype.push(avgAction);
      }
    }

    return prototype;
  }

  private inferPreconditions(behaviors: Behavior[]): SkillPrecondition {
    const initialStates = behaviors.map(b => b.sequence[0]?.state || []).filter(s => s.length > 0);

    if (initialStates.length === 0) {
      return {
        stateRange: { min: [], max: [] },
        requiredSymbols: [],
        minConfidence: 0.5,
      };
    }

    const dim = initialStates[0].length;
    const min = new Array(dim).fill(Infinity);
    const max = new Array(dim).fill(-Infinity);

    for (const state of initialStates) {
      for (let i = 0; i < dim; i++) {
        min[i] = Math.min(min[i], state[i] || 0);
        max[i] = Math.max(max[i], state[i] || 0);
      }
    }

    const initialSymbols = behaviors
      .map(b => b.sequence[0]?.symbol)
      .filter(Boolean);

    const symbolCounts: Map<string, number> = new Map();
    for (const sym of initialSymbols) {
      symbolCounts.set(sym, (symbolCounts.get(sym) || 0) + 1);
    }

    const requiredSymbols = Array.from(symbolCounts.entries())
      .filter(([_, count]) => count / behaviors.length > 0.5)
      .map(([sym]) => sym);

    return {
      stateRange: { min, max },
      requiredSymbols,
      minConfidence: 0.5,
    };
  }

  private inferEffects(behaviors: Behavior[]): SkillEffects {
    const stateChanges: number[][] = [];
    const rewards: number[] = [];
    const durations: number[] = [];

    for (const behavior of behaviors) {
      if (behavior.sequence.length >= 2) {
        const initial = behavior.sequence[0].state;
        const final = behavior.sequence[behavior.sequence.length - 1].state;
        stateChanges.push(final.map((v, i) => v - (initial[i] || 0)));
      }
      rewards.push(behavior.outcome.totalReward);
      durations.push(behavior.outcome.duration);
    }

    const expectedStateChange = stateChanges.length > 0 ? this.averageVectors(stateChanges) : [];
    const expectedReward = rewards.reduce((a, b) => a + b, 0) / (rewards.length || 1);
    const expectedDuration = durations.reduce((a, b) => a + b, 0) / (durations.length || 1);

    return {
      expectedStateChange,
      expectedReward,
      expectedDuration,
      sideEffects: [],
    };
  }

  private averageVectors(vectors: number[][]): number[] {
    if (vectors.length === 0) return [];
    const dim = Math.max(...vectors.map(v => v.length));
    const avg = new Array(dim).fill(0);

    for (const vec of vectors) {
      for (let i = 0; i < vec.length; i++) {
        avg[i] += vec[i] / vectors.length;
      }
    }

    return avg;
  }

  private computeComposability(symbolPattern: string[]): number {
    const uniqueSymbols = new Set(symbolPattern).size;
    const patternLength = symbolPattern.length;

    if (patternLength === 0) return 0;

    const diversity = uniqueSymbols / patternLength;
    const lengthBonus = Math.min(patternLength / 10, 1);

    return (diversity * 0.6 + lengthBonus * 0.4);
  }

  private computeAbstractionLevel(behaviors: Behavior[]): number {
    const avgLength = behaviors.reduce((s, b) => s + b.sequence.length, 0) / behaviors.length;
    return Math.min(Math.log2(avgLength + 1) / 5, 1);
  }

  private generateDescription(
    symbolPattern: string[],
    effects: SkillEffects,
    successRate: number
  ): string {
    const patternStr = symbolPattern.slice(0, 3).join(" → ");
    const rewardStr = effects.expectedReward >= 0 ? "positive" : "negative";

    return `Skill pattern [${patternStr}] with ${rewardStr} reward (${(successRate * 100).toFixed(0)}% success)`;
  }
}

// ==================== SKILL NAME GENERATOR ====================

class SkillNameGenerator {
  private prefixes = ["reach", "grasp", "move", "push", "pull", "rotate", "lift", "place", "navigate", "acquire"];
  private modifiers = ["quick", "precise", "efficient", "stable", "adaptive", "robust"];
  private suffixes = ["action", "maneuver", "operation", "procedure", "routine"];

  generate(symbolPattern: string[], effects: SkillEffects, domain: string): string {
    const prefix = this.selectPrefix(effects);
    const modifier = this.selectModifier(symbolPattern);
    const suffix = this.selectSuffix(symbolPattern.length);

    return `${prefix}_${modifier}_${suffix}_${domain.slice(0, 3)}`;
  }

  private selectPrefix(effects: SkillEffects): string {
    if (effects.expectedReward > 0.5) {
      return this.prefixes[Math.floor(Math.random() * 5)];
    }
    return this.prefixes[5 + Math.floor(Math.random() * 5)];
  }

  private selectModifier(pattern: string[]): string {
    const complexity = pattern.length;
    if (complexity <= 3) return "quick";
    if (complexity <= 6) return "efficient";
    return "precise";
  }

  private selectSuffix(patternLength: number): string {
    if (patternLength <= 2) return "action";
    if (patternLength <= 5) return "maneuver";
    return "procedure";
  }
}

// ==================== SKILL LIBRARY ====================

export class SkillLibrary {
  private skills: Map<string, DiscoveredSkill> = new Map();
  private domainIndex: Map<string, string[]> = new Map();
  private symbolIndex: Map<string, string[]> = new Map();

  add(skill: DiscoveredSkill): void {
    this.skills.set(skill.id, skill);

    for (const domain of skill.domains) {
      const domainSkills = this.domainIndex.get(domain) || [];
      domainSkills.push(skill.id);
      this.domainIndex.set(domain, domainSkills);
    }

    for (const symbol of skill.symbolPattern) {
      const symbolSkills = this.symbolIndex.get(symbol) || [];
      symbolSkills.push(skill.id);
      this.symbolIndex.set(symbol, symbolSkills);
    }
  }

  getSkill(id: string): DiscoveredSkill | undefined {
    return this.skills.get(id);
  }

  getByDomain(domain: string): DiscoveredSkill[] {
    const ids = this.domainIndex.get(domain) || [];
    return ids.map(id => this.skills.get(id)!).filter(Boolean);
  }

  getBySymbol(symbol: string): DiscoveredSkill[] {
    const ids = this.symbolIndex.get(symbol) || [];
    return ids.map(id => this.skills.get(id)!).filter(Boolean);
  }

  findApplicable(state: number[], domain?: string): DiscoveredSkill[] {
    let candidates = Array.from(this.skills.values());

    if (domain) {
      candidates = candidates.filter(s => s.domains.includes(domain));
    }

    return candidates.filter(skill => this.checkPreconditions(skill, state));
  }

  private checkPreconditions(skill: DiscoveredSkill, state: number[]): boolean {
    const { min, max } = skill.preconditions.stateRange;

    for (let i = 0; i < Math.min(state.length, min.length); i++) {
      if (state[i] < min[i] - 0.1 || state[i] > max[i] + 0.1) {
        return false;
      }
    }

    return true;
  }

  recordUsage(skillId: string, success: boolean, reward: number): void {
    const skill = this.skills.get(skillId);
    if (!skill) return;

    skill.usageCount++;
    skill.lastUsed = Date.now();

    const alpha = 1 / skill.usageCount;
    skill.successRate = skill.successRate * (1 - alpha) + (success ? 1 : 0) * alpha;
  }

  getTopSkills(n = 10): DiscoveredSkill[] {
    return Array.from(this.skills.values())
      .sort((a, b) => {
        const scoreA = a.successRate * Math.log(a.usageCount + 1);
        const scoreB = b.successRate * Math.log(b.usageCount + 1);
        return scoreB - scoreA;
      })
      .slice(0, n);
  }

  size(): number {
    return this.skills.size;
  }

  getStats(): Record<string, unknown> {
    const skills = Array.from(this.skills.values());
    return {
      totalSkills: skills.length,
      avgSuccessRate: skills.reduce((s, sk) => s + sk.successRate, 0) / (skills.length || 1),
      avgUsageCount: skills.reduce((s, sk) => s + sk.usageCount, 0) / (skills.length || 1),
      domains: Array.from(this.domainIndex.keys()),
    };
  }
}

// ==================== SKILL DISCOVERY ENGINE ====================

export class SkillDiscoveryEngine {
  behaviorBuffer: BehaviorBuffer;
  encoder: BehaviorEncoder;
  clusterer: SkillClusterer;
  synthesizer: SkillSynthesizer;
  library: SkillLibrary;

  private discoveryInterval: number;
  private lastDiscovery: number = 0;
  private minBehaviorsForDiscovery: number = 10;

  constructor(config: {
    maxBehaviors?: number;
    minClusterSize?: number;
    similarityThreshold?: number;
    discoveryInterval?: number;
  } = {}) {
    this.behaviorBuffer = new BehaviorBuffer(config.maxBehaviors || 10000);
    this.encoder = new BehaviorEncoder();
    this.clusterer = new SkillClusterer(
      config.minClusterSize || 3,
      config.similarityThreshold || 0.7
    );
    this.synthesizer = new SkillSynthesizer();
    this.library = new SkillLibrary();
    this.discoveryInterval = config.discoveryInterval || 1000 * 60;
  }

  recordBehavior(
    sequence: BehaviorStep[],
    context: number[],
    outcome: BehaviorOutcome,
    domain: string
  ): string {
    return this.behaviorBuffer.add({ sequence, context, outcome, domain });
  }

  async discover(domain?: string): Promise<DiscoveredSkill[]> {
    const now = Date.now();
    if (now - this.lastDiscovery < this.discoveryInterval) {
      return [];
    }

    this.lastDiscovery = now;

    const behaviors = domain
      ? this.behaviorBuffer.getByDomain(domain)
      : this.behaviorBuffer.getSuccessfulBehaviors(0);

    if (behaviors.length < this.minBehaviorsForDiscovery) {
      return [];
    }

    const encoded = behaviors.map(b => ({
      id: b.id,
      encoding: this.encoder.encode(b),
    }));

    const clusters = this.clusterer.cluster(encoded);

    const newSkills: DiscoveredSkill[] = [];

    for (const cluster of clusters) {
      const clusterBehaviors = cluster.members
        .map(id => this.behaviorBuffer.getBehavior(id))
        .filter((b): b is Behavior => b !== undefined);

      if (clusterBehaviors.length === 0) continue;

      const clusterDomain = domain || clusterBehaviors[0].domain;
      const skill = this.synthesizer.synthesize(cluster, clusterBehaviors, clusterDomain);

      const isDuplicate = this.isDuplicateSkill(skill);
      if (!isDuplicate) {
        this.library.add(skill);
        newSkills.push(skill);
      }
    }

    return newSkills;
  }

  private isDuplicateSkill(newSkill: DiscoveredSkill): boolean {
    const existingSkills = this.library.getByDomain(newSkill.domains[0]);

    for (const existing of existingSkills) {
      const patternOverlap = this.computePatternOverlap(
        newSkill.symbolPattern,
        existing.symbolPattern
      );
      if (patternOverlap > 0.8) return true;
    }

    return false;
  }

  private computePatternOverlap(p1: string[], p2: string[]): number {
    const set1 = new Set(p1);
    const set2 = new Set(p2);
    const intersection = [...set1].filter(x => set2.has(x)).length;
    const union = new Set([...p1, ...p2]).size;
    return intersection / union;
  }

  getApplicableSkills(state: number[], domain?: string): DiscoveredSkill[] {
    return this.library.findApplicable(state, domain);
  }

  useSkill(skillId: string, success: boolean, reward: number): void {
    this.library.recordUsage(skillId, success, reward);
  }

  getStats(): Record<string, unknown> {
    return {
      behaviorBufferSize: this.behaviorBuffer.size(),
      libraryStats: this.library.getStats(),
      lastDiscovery: this.lastDiscovery,
    };
  }
}
