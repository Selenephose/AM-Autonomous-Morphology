/**
 * Curiosity and Novelty Reward System
 *
 * Implements:
 * - Information gain rewards
 * - Model uncertainty reduction
 * - Concept formation bonuses
 * - Abstraction creation incentives
 */

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";

// ==================== TYPES ====================

export interface NoveltySignal {
  informationGain: number;
  predictionError: number;
  stateNovelty: number;
  conceptNovelty: number;
  totalIntrinsic: number;
}

export interface CuriosityState {
  explorationBonus: number;
  uncertaintyReduction: number;
  noveltyHistory: number[];
  currentFocus: string;
}

export interface ConceptFormation {
  id: string;
  embedding: number[];
  memberStates: number[][];
  coherence: number;
  noveltyScore: number;
  createdAt: number;
}

// ==================== RANDOM NETWORK DISTILLATION ====================

export class RandomNetworkDistillation {
  private targetNetwork: tf.LayersModel;
  private predictorNetwork: tf.LayersModel;
  private inputDim: number;
  private featureDim: number;

  constructor(inputDim = 64, featureDim = 32) {
    this.inputDim = inputDim;
    this.featureDim = featureDim;
    this.targetNetwork = this.buildFixedNetwork();
    this.predictorNetwork = this.buildTrainableNetwork();
  }

  private buildFixedNetwork(): tf.LayersModel {
    const input = tf.input({ shape: [this.inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu", trainable: false }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu", trainable: false }).apply(h1) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: this.featureDim, trainable: false }).apply(h2) as tf.SymbolicTensor;

    return tf.model({ inputs: input, outputs: output });
  }

  private buildTrainableNetwork(): tf.LayersModel {
    const input = tf.input({ shape: [this.inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: this.featureDim }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({ optimizer: tf.train.adam(1e-4), loss: "meanSquaredError" });
    return model;
  }

  computeNovelty(state: number[]): number {
    return tf.tidy(() => {
      const paddedState = this.padToLength(state, this.inputDim);
      const inputT = tf.tensor2d([paddedState]);

      const targetFeatures = this.targetNetwork.predict(inputT) as tf.Tensor;
      const predictedFeatures = this.predictorNetwork.predict(inputT) as tf.Tensor;

      const error = targetFeatures.sub(predictedFeatures).square().mean();
      return error.dataSync()[0];
    });
  }

  async train(states: number[][]): Promise<number> {
    if (states.length === 0) return 0;

    const paddedStates = states.map(s => this.padToLength(s, this.inputDim));
    const inputT = tf.tensor2d(paddedStates);

    const targetFeatures = this.targetNetwork.predict(inputT) as tf.Tensor;

    const result = await this.predictorNetwork.fit(inputT, targetFeatures, {
      epochs: 1,
      verbose: 0,
    });

    inputT.dispose();
    targetFeatures.dispose();

    return result.history.loss[0] as number;
  }

  private padToLength(arr: number[], length: number): number[] {
    if (arr.length >= length) return arr.slice(0, length);
    return [...arr, ...new Array(length - arr.length).fill(0)];
  }
}

// ==================== INFORMATION GAIN ESTIMATOR ====================

export class InformationGainEstimator {
  private worldModelPrior: Map<string, number[]> = new Map();
  private worldModelPosterior: Map<string, number[]> = new Map();
  private observationCount: number = 0;

  estimateGain(state: number[], prediction: number[], actual: number[]): number {
    const stateKey = this.quantizeState(state);

    const priorEntropy = this.computeEntropy(this.worldModelPrior.get(stateKey) || [0.5, 0.5]);

    this.updatePosterior(stateKey, prediction, actual);

    const posteriorEntropy = this.computeEntropy(this.worldModelPosterior.get(stateKey) || [0.5, 0.5]);

    const informationGain = Math.max(0, priorEntropy - posteriorEntropy);

    this.worldModelPrior.set(stateKey, this.worldModelPosterior.get(stateKey) || []);

    return informationGain;
  }

  private quantizeState(state: number[]): string {
    return state.map(v => Math.round(v * 10) / 10).join(",");
  }

  private updatePosterior(stateKey: string, prediction: number[], actual: number[]): void {
    const error = actual.map((a, i) => Math.abs(a - (prediction[i] || 0)));
    const avgError = error.reduce((s, e) => s + e, 0) / error.length;

    const accuracy = 1 - Math.min(avgError, 1);
    const posterior = [accuracy, 1 - accuracy];

    this.worldModelPosterior.set(stateKey, posterior);
  }

  private computeEntropy(distribution: number[]): number {
    const sum = distribution.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const probs = distribution.map(p => Math.abs(p) / sum);

    return -probs.reduce((h, p) => {
      if (p > 0) {
        return h + p * Math.log2(p);
      }
      return h;
    }, 0);
  }

  getStateUncertainty(state: number[]): number {
    const stateKey = this.quantizeState(state);
    const posterior = this.worldModelPosterior.get(stateKey);

    if (!posterior) return 1.0;
    return this.computeEntropy(posterior);
  }
}

// ==================== CONCEPT FORMATION ENGINE ====================

export class ConceptFormationEngine {
  private concepts: Map<string, ConceptFormation> = new Map();
  private stateBuffer: number[][] = [];
  private bufferSize: number = 1000;
  private conceptThreshold: number = 0.6;

  addObservation(state: number[]): void {
    this.stateBuffer.push(state);
    if (this.stateBuffer.length > this.bufferSize) {
      this.stateBuffer = this.stateBuffer.slice(-this.bufferSize);
    }
  }

  formConcepts(): ConceptFormation[] {
    if (this.stateBuffer.length < 10) return [];

    const newConcepts: ConceptFormation[] = [];
    const clusters = this.clusterStates();

    for (const cluster of clusters) {
      if (cluster.length < 3) continue;

      const embedding = this.computeCentroid(cluster);
      const coherence = this.computeCoherence(cluster, embedding);

      if (coherence < this.conceptThreshold) continue;

      const novelty = this.computeConceptNovelty(embedding);

      if (novelty > 0.3) {
        const concept: ConceptFormation = {
          id: `concept_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
          embedding,
          memberStates: cluster,
          coherence,
          noveltyScore: novelty,
          createdAt: Date.now(),
        };

        this.concepts.set(concept.id, concept);
        newConcepts.push(concept);
      }
    }

    return newConcepts;
  }

  private clusterStates(): number[][][] {
    const clusters: number[][][] = [];
    const assigned = new Set<number>();

    for (let i = 0; i < this.stateBuffer.length; i++) {
      if (assigned.has(i)) continue;

      const cluster: number[][] = [this.stateBuffer[i]];
      assigned.add(i);

      for (let j = i + 1; j < this.stateBuffer.length; j++) {
        if (assigned.has(j)) continue;

        const sim = this.cosineSimilarity(this.stateBuffer[i], this.stateBuffer[j]);
        if (sim > 0.7) {
          cluster.push(this.stateBuffer[j]);
          assigned.add(j);
        }
      }

      if (cluster.length >= 3) {
        clusters.push(cluster);
      }
    }

    return clusters;
  }

  private computeCentroid(states: number[][]): number[] {
    if (states.length === 0) return [];
    const dim = states[0].length;
    const centroid = new Array(dim).fill(0);

    for (const state of states) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += (state[i] || 0) / states.length;
      }
    }

    return centroid;
  }

  private computeCoherence(states: number[][], centroid: number[]): number {
    let totalSim = 0;

    for (const state of states) {
      totalSim += this.cosineSimilarity(state, centroid);
    }

    return totalSim / states.length;
  }

  private computeConceptNovelty(embedding: number[]): number {
    if (this.concepts.size === 0) return 1.0;

    let minDistance = Infinity;

    for (const concept of this.concepts.values()) {
      const sim = this.cosineSimilarity(embedding, concept.embedding);
      const distance = 1 - sim;
      minDistance = Math.min(minDistance, distance);
    }

    return minDistance;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * (b[i] || 0), 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  getConceptNoveltyBonus(state: number[]): number {
    if (this.concepts.size === 0) return 0;

    let noveltyBonus = 0;

    for (const concept of this.concepts.values()) {
      const sim = this.cosineSimilarity(state, concept.embedding);
      if (sim > 0.8) {
        noveltyBonus += concept.noveltyScore * 0.1;
      }
    }

    return Math.min(noveltyBonus, 0.5);
  }

  getConcepts(): ConceptFormation[] {
    return Array.from(this.concepts.values());
  }
}

// ==================== ABSTRACTION REWARD ====================

export class AbstractionReward {
  private abstractionHistory: Array<{ level: number; timestamp: number }> = [];
  private currentAbstractionLevel: number = 0;

  computeReward(
    currentSymbols: string[],
    previousSymbols: string[],
    hierarchyDepth: number
  ): number {
    let reward = 0;

    const symbolCompression = this.computeSymbolCompression(currentSymbols, previousSymbols);
    reward += symbolCompression * 0.3;

    const hierarchyBonus = Math.log(hierarchyDepth + 1) * 0.2;
    reward += hierarchyBonus;

    const abstractionIncrease = hierarchyDepth - this.currentAbstractionLevel;
    if (abstractionIncrease > 0) {
      reward += abstractionIncrease * 0.5;
      this.currentAbstractionLevel = hierarchyDepth;
    }

    this.abstractionHistory.push({
      level: hierarchyDepth,
      timestamp: Date.now(),
    });

    return reward;
  }

  private computeSymbolCompression(current: string[], previous: string[]): number {
    if (previous.length === 0) return 0;

    const currentUnique = new Set(current).size;
    const previousUnique = new Set(previous).size;

    if (previousUnique === 0) return 0;

    const compression = 1 - (currentUnique / previousUnique);
    return Math.max(0, compression);
  }

  getAbstractionTrend(): number {
    if (this.abstractionHistory.length < 2) return 0;

    const recent = this.abstractionHistory.slice(-10);
    const older = this.abstractionHistory.slice(-20, -10);

    if (older.length === 0) return 0;

    const recentAvg = recent.reduce((s, h) => s + h.level, 0) / recent.length;
    const olderAvg = older.reduce((s, h) => s + h.level, 0) / older.length;

    return recentAvg - olderAvg;
  }
}

// ==================== CURIOSITY ENGINE ====================

export class CuriosityEngine {
  rnd: RandomNetworkDistillation;
  informationGain: InformationGainEstimator;
  conceptFormation: ConceptFormationEngine;
  abstractionReward: AbstractionReward;

  private noveltyHistory: number[] = [];
  private intrinsicWeight: number = 0.5;
  private decayRate: number = 0.995;

  constructor(inputDim = 64) {
    this.rnd = new RandomNetworkDistillation(inputDim);
    this.informationGain = new InformationGainEstimator();
    this.conceptFormation = new ConceptFormationEngine();
    this.abstractionReward = new AbstractionReward();
  }

  computeIntrinsicReward(
    state: number[],
    prediction: number[],
    actual: number[],
    symbols: string[],
    previousSymbols: string[],
    hierarchyDepth: number
  ): NoveltySignal {
    const stateNovelty = this.rnd.computeNovelty(state);

    const infoGain = this.informationGain.estimateGain(state, prediction, actual);

    this.conceptFormation.addObservation(state);
    const conceptNovelty = this.conceptFormation.getConceptNoveltyBonus(state);

    const abstractionBonus = this.abstractionReward.computeReward(
      symbols,
      previousSymbols,
      hierarchyDepth
    );

    const predictionError = actual.reduce((sum, a, i) =>
      sum + Math.pow(a - (prediction[i] || 0), 2), 0
    ) / actual.length;

    const totalIntrinsic = this.computeTotal(
      stateNovelty,
      infoGain,
      conceptNovelty,
      abstractionBonus,
      predictionError
    );

    this.noveltyHistory.push(totalIntrinsic);
    if (this.noveltyHistory.length > 1000) {
      this.noveltyHistory = this.noveltyHistory.slice(-500);
    }

    return {
      informationGain: infoGain,
      predictionError,
      stateNovelty,
      conceptNovelty,
      totalIntrinsic,
    };
  }

  private computeTotal(
    stateNovelty: number,
    infoGain: number,
    conceptNovelty: number,
    abstractionBonus: number,
    predictionError: number
  ): number {
    const avgNovelty = this.noveltyHistory.length > 0
      ? this.noveltyHistory.reduce((a, b) => a + b, 0) / this.noveltyHistory.length
      : 0.5;

    const normalizedState = stateNovelty / (avgNovelty + 0.01);
    const normalizedError = Math.tanh(predictionError);

    const total =
      normalizedState * 0.3 +
      infoGain * 0.25 +
      conceptNovelty * 0.2 +
      abstractionBonus * 0.15 +
      normalizedError * 0.1;

    return total * this.intrinsicWeight;
  }

  async train(states: number[][]): Promise<void> {
    await this.rnd.train(states);
  }

  formConcepts(): ConceptFormation[] {
    return this.conceptFormation.formConcepts();
  }

  getCuriosityState(): CuriosityState {
    const recentNovelty = this.noveltyHistory.slice(-10);
    const avgNovelty = recentNovelty.reduce((a, b) => a + b, 0) / (recentNovelty.length || 1);

    return {
      explorationBonus: avgNovelty,
      uncertaintyReduction: this.abstractionReward.getAbstractionTrend(),
      noveltyHistory: this.noveltyHistory.slice(-100),
      currentFocus: avgNovelty > 0.5 ? "exploration" : "exploitation",
    };
  }

  setIntrinsicWeight(weight: number): void {
    this.intrinsicWeight = Math.max(0, Math.min(1, weight));
  }

  decay(): void {
    this.intrinsicWeight *= this.decayRate;
    this.intrinsicWeight = Math.max(0.1, this.intrinsicWeight);
  }

  getStats(): Record<string, unknown> {
    return {
      intrinsicWeight: this.intrinsicWeight,
      avgNovelty: this.noveltyHistory.reduce((a, b) => a + b, 0) / (this.noveltyHistory.length || 1),
      conceptCount: this.conceptFormation.getConcepts().length,
      abstractionTrend: this.abstractionReward.getAbstractionTrend(),
    };
  }
}
