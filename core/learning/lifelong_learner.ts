/**
 * Lifelong Learning System with Transfer Capabilities
 *
 * Enables:
 * - Continual learning without catastrophic forgetting
 * - Domain transfer via shared representations
 * - Progressive complexity through curriculum
 * - Knowledge consolidation and replay
 */

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";

// ==================== TYPES ====================

export interface DomainKnowledge {
  id: string;
  name: string;
  embedding: number[];
  taskCount: number;
  successRate: number;
  transferSources: string[];
  transferTargets: string[];
  competence: number;
  lastUpdated: number;
  createdAt: number;
}

export interface TransferResult {
  sourceDomain: string;
  targetDomain: string;
  transferability: number;
  performanceGain: number;
  featuresTransferred: number;
}

export interface LearningProgress {
  domain: string;
  taskId: string;
  epoch: number;
  loss: number;
  accuracy: number;
  timestamp: number;
}

export interface CurriculumTask {
  id: string;
  domain: string;
  difficulty: number;
  prerequisites: string[];
  completed: boolean;
  attempts: number;
  bestScore: number;
}

// ==================== ELASTIC WEIGHT CONSOLIDATION ====================

export class ElasticWeightConsolidation {
  private fisherMatrices: Map<string, number[][]> = new Map();
  private optimalWeights: Map<string, number[][]> = new Map();
  private lambda: number;

  constructor(lambda = 400) {
    this.lambda = lambda;
  }

  async computeFisher(
    model: tf.LayersModel,
    data: tf.Tensor,
    labels: tf.Tensor,
    taskId: string
  ): Promise<void> {
    const weights = model.getWeights();
    const fisherDiagonals: number[][] = [];

    for (let layerIdx = 0; layerIdx < weights.length; layerIdx++) {
      const weightShape = weights[layerIdx].shape;
      const flatSize = weightShape.reduce((a, b) => a * b, 1);
      const diagonal = new Array(flatSize).fill(0);

      const numSamples = Math.min(data.shape[0], 100);
      for (let i = 0; i < numSamples; i++) {
        const sample = data.slice([i], [1]);
        const label = labels.slice([i], [1]);

        const gradFunc = () => {
          const pred = model.predict(sample) as tf.Tensor;
          return tf.losses.softmaxCrossEntropy(label, pred);
        };

        const grads = tf.grads(gradFunc);
        const gradValues = grads([weights[layerIdx]]);

        if (gradValues[0]) {
          const gradFlat = Array.from(gradValues[0].dataSync());
          for (let j = 0; j < flatSize && j < gradFlat.length; j++) {
            diagonal[j] += gradFlat[j] * gradFlat[j] / numSamples;
          }
          gradValues[0].dispose();
        }
      }

      fisherDiagonals.push(diagonal);
    }

    this.fisherMatrices.set(taskId, fisherDiagonals);
    this.optimalWeights.set(taskId, weights.map(w => Array.from(w.dataSync())));
  }

  computePenalty(model: tf.LayersModel): number {
    let totalPenalty = 0;
    const currentWeights = model.getWeights();

    for (const [taskId, fisher] of this.fisherMatrices) {
      const optimal = this.optimalWeights.get(taskId)!;

      for (let layerIdx = 0; layerIdx < currentWeights.length && layerIdx < fisher.length; layerIdx++) {
        const currentFlat = Array.from(currentWeights[layerIdx].dataSync());
        const optimalFlat = optimal[layerIdx];
        const fisherDiag = fisher[layerIdx];

        for (let i = 0; i < currentFlat.length && i < optimalFlat.length && i < fisherDiag.length; i++) {
          const diff = currentFlat[i] - optimalFlat[i];
          totalPenalty += fisherDiag[i] * diff * diff;
        }
      }
    }

    return this.lambda * totalPenalty * 0.5;
  }

  getTaskCount(): number {
    return this.fisherMatrices.size;
  }
}

// ==================== PROGRESSIVE NEURAL NETWORK ====================

export class ProgressiveNeuralNetwork {
  private columns: tf.LayersModel[] = [];
  private lateralConnections: Map<string, tf.LayersModel> = new Map();
  private domainToColumn: Map<string, number> = new Map();

  private inputDim: number;
  private hiddenDim: number;
  private outputDim: number;

  constructor(inputDim: number, hiddenDim: number, outputDim: number) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
  }

  addColumn(domain: string): number {
    const columnIdx = this.columns.length;

    const input = tf.input({ shape: [this.inputDim] });
    let h: tf.SymbolicTensor = tf.layers.dense({ units: this.hiddenDim, activation: "relu" }).apply(input) as tf.SymbolicTensor;

    if (columnIdx > 0) {
      const lateralInputs: tf.SymbolicTensor[] = [h];

      for (let i = 0; i < columnIdx; i++) {
        const lateralInput = tf.input({ shape: [this.hiddenDim], name: `lateral_${i}` });
        const lateralH = tf.layers.dense({ units: this.hiddenDim, activation: "relu" }).apply(lateralInput) as tf.SymbolicTensor;
        lateralInputs.push(lateralH);
      }

      h = tf.layers.add().apply(lateralInputs) as tf.SymbolicTensor;
    }

    h = tf.layers.dense({ units: this.hiddenDim, activation: "relu" }).apply(h) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: this.outputDim, activation: "softmax" }).apply(h) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });

    this.columns.push(model);
    this.domainToColumn.set(domain, columnIdx);

    return columnIdx;
  }

  getColumn(domain: string): tf.LayersModel | null {
    const idx = this.domainToColumn.get(domain);
    return idx !== undefined ? this.columns[idx] : null;
  }

  async predict(domain: string, input: number[]): Promise<number[]> {
    const column = this.getColumn(domain);
    if (!column) return [];

    return tf.tidy(() => {
      const inputT = tf.tensor2d([input]);
      const output = column.predict(inputT) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  getColumnCount(): number {
    return this.columns.length;
  }

  getDomains(): string[] {
    return Array.from(this.domainToColumn.keys());
  }
}

// ==================== DOMAIN TRANSFER MANAGER ====================

export class DomainTransferManager {
  private domains: Map<string, DomainKnowledge> = new Map();
  private transferMatrix: Map<string, Map<string, number>> = new Map();
  private sharedEncoder: tf.LayersModel;
  private domainEmbeddingDim: number;

  constructor(inputDim = 64, embeddingDim = 32) {
    this.domainEmbeddingDim = embeddingDim;
    this.sharedEncoder = this.buildSharedEncoder(inputDim, embeddingDim);
  }

  private buildSharedEncoder(inputDim: number, embeddingDim: number): tf.LayersModel {
    const input = tf.input({ shape: [inputDim] });
    const h1 = tf.layers.dense({ units: 128, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1) as tf.SymbolicTensor;
    const embedding = tf.layers.dense({ units: embeddingDim }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: embedding });
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    return model;
  }

  registerDomain(name: string, sampleData?: number[][]): string {
    const id = `domain_${name}_${Date.now()}`;

    let embedding = new Array(this.domainEmbeddingDim).fill(0);
    if (sampleData && sampleData.length > 0) {
      embedding = this.computeDomainEmbedding(sampleData);
    }

    const domain: DomainKnowledge = {
      id,
      name,
      embedding,
      taskCount: 0,
      successRate: 0,
      transferSources: [],
      transferTargets: [],
      competence: 0,
      lastUpdated: Date.now(),
      createdAt: Date.now(),
    };

    this.domains.set(id, domain);
    this.transferMatrix.set(id, new Map());

    this.updateTransferabilities(id);

    return id;
  }

  private computeDomainEmbedding(samples: number[][]): number[] {
    if (samples.length === 0) return new Array(this.domainEmbeddingDim).fill(0);

    return tf.tidy(() => {
      const padded = samples.map(s => {
        if (s.length < 64) return [...s, ...new Array(64 - s.length).fill(0)];
        return s.slice(0, 64);
      });

      const inputT = tf.tensor2d(padded);
      const embedding = this.sharedEncoder.predict(inputT) as tf.Tensor;
      const mean = embedding.mean(0);
      return Array.from(mean.dataSync());
    });
  }

  private updateTransferabilities(newDomainId: string): void {
    const newDomain = this.domains.get(newDomainId);
    if (!newDomain) return;

    for (const [otherId, otherDomain] of this.domains) {
      if (otherId === newDomainId) continue;

      const similarity = this.cosineSimilarity(newDomain.embedding, otherDomain.embedding);
      const transferability = Math.max(0, similarity);

      this.transferMatrix.get(newDomainId)!.set(otherId, transferability);
      this.transferMatrix.get(otherId)!.set(newDomainId, transferability);

      if (transferability > 0.5) {
        newDomain.transferSources.push(otherId);
        otherDomain.transferTargets.push(newDomainId);
      }
    }
  }

  async transfer(sourceDomainId: string, targetDomainId: string): Promise<TransferResult> {
    const source = this.domains.get(sourceDomainId);
    const target = this.domains.get(targetDomainId);

    if (!source || !target) {
      return {
        sourceDomain: sourceDomainId,
        targetDomain: targetDomainId,
        transferability: 0,
        performanceGain: 0,
        featuresTransferred: 0,
      };
    }

    const transferability = this.transferMatrix.get(sourceDomainId)?.get(targetDomainId) || 0;

    const performanceGain = transferability * source.competence * 0.5;
    target.competence = Math.min(1, target.competence + performanceGain);

    const featuresTransferred = Math.floor(this.domainEmbeddingDim * transferability);
    for (let i = 0; i < featuresTransferred; i++) {
      target.embedding[i] = target.embedding[i] * 0.7 + source.embedding[i] * 0.3;
    }

    target.lastUpdated = Date.now();

    return {
      sourceDomain: source.name,
      targetDomain: target.name,
      transferability,
      performanceGain,
      featuresTransferred,
    };
  }

  findBestTransferSource(targetDomainId: string): string | null {
    const transferabilities = this.transferMatrix.get(targetDomainId);
    if (!transferabilities) return null;

    let bestSource: string | null = null;
    let bestScore = 0;

    for (const [sourceId, transferability] of transferabilities) {
      const source = this.domains.get(sourceId);
      if (!source) continue;

      const score = transferability * source.competence;
      if (score > bestScore) {
        bestScore = score;
        bestSource = sourceId;
      }
    }

    return bestSource;
  }

  updateCompetence(domainId: string, taskSuccess: boolean, performanceScore: number): void {
    const domain = this.domains.get(domainId);
    if (!domain) return;

    domain.taskCount++;
    const alpha = 1 / domain.taskCount;
    domain.successRate = domain.successRate * (1 - alpha) + (taskSuccess ? 1 : 0) * alpha;
    domain.competence = domain.competence * 0.9 + performanceScore * 0.1;
    domain.lastUpdated = Date.now();
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  getDomainKnowledge(domainId: string): DomainKnowledge | undefined {
    return this.domains.get(domainId);
  }

  getAllDomains(): DomainKnowledge[] {
    return Array.from(this.domains.values());
  }

  getTransferMatrix(): Map<string, Map<string, number>> {
    return this.transferMatrix;
  }
}

// ==================== CURRICULUM MANAGER ====================

export class CurriculumManager {
  private tasks: Map<string, CurriculumTask> = new Map();
  private completedTasks: string[] = [];
  private currentDifficulty: number = 0;
  private difficultyIncrement: number = 0.1;
  private maxDifficulty: number = 1.0;

  addTask(task: Omit<CurriculumTask, "completed" | "attempts" | "bestScore">): string {
    const fullTask: CurriculumTask = {
      ...task,
      completed: false,
      attempts: 0,
      bestScore: 0,
    };
    this.tasks.set(task.id, fullTask);
    return task.id;
  }

  getNextTask(domain?: string): CurriculumTask | null {
    const availableTasks = Array.from(this.tasks.values())
      .filter(t => !t.completed)
      .filter(t => !domain || t.domain === domain)
      .filter(t => t.difficulty <= this.currentDifficulty + 0.2)
      .filter(t => t.prerequisites.every(p => this.completedTasks.includes(p)));

    if (availableTasks.length === 0) {
      if (this.currentDifficulty < this.maxDifficulty) {
        this.currentDifficulty += this.difficultyIncrement;
        return this.getNextTask(domain);
      }
      return null;
    }

    availableTasks.sort((a, b) => {
      const diffA = Math.abs(a.difficulty - this.currentDifficulty);
      const diffB = Math.abs(b.difficulty - this.currentDifficulty);
      return diffA - diffB;
    });

    return availableTasks[0];
  }

  recordAttempt(taskId: string, score: number): { completed: boolean; newDifficulty: number } {
    const task = this.tasks.get(taskId);
    if (!task) return { completed: false, newDifficulty: this.currentDifficulty };

    task.attempts++;
    task.bestScore = Math.max(task.bestScore, score);

    const threshold = 0.7 + task.difficulty * 0.2;

    if (score >= threshold) {
      task.completed = true;
      this.completedTasks.push(taskId);

      if (this.shouldIncreaseDifficulty()) {
        this.currentDifficulty = Math.min(this.maxDifficulty, this.currentDifficulty + this.difficultyIncrement);
      }
    }

    return { completed: task.completed, newDifficulty: this.currentDifficulty };
  }

  private shouldIncreaseDifficulty(): boolean {
    const recentTasks = this.completedTasks.slice(-5);
    if (recentTasks.length < 3) return false;

    const recentScores = recentTasks
      .map(id => this.tasks.get(id)?.bestScore || 0);

    const avgScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;
    return avgScore > 0.85;
  }

  getCurrentDifficulty(): number {
    return this.currentDifficulty;
  }

  getProgress(): { completed: number; total: number; currentDifficulty: number } {
    return {
      completed: this.completedTasks.length,
      total: this.tasks.size,
      currentDifficulty: this.currentDifficulty,
    };
  }

  getTasksByDomain(domain: string): CurriculumTask[] {
    return Array.from(this.tasks.values()).filter(t => t.domain === domain);
  }
}

// ==================== EXPERIENCE REPLAY BUFFER ====================

export class PrioritizedExperienceReplay {
  private buffer: Array<{
    experience: any;
    priority: number;
    domain: string;
    timestamp: number;
  }> = [];
  private maxSize: number;
  private alpha: number;
  private beta: number;

  constructor(maxSize = 100000, alpha = 0.6, beta = 0.4) {
    this.maxSize = maxSize;
    this.alpha = alpha;
    this.beta = beta;
  }

  add(experience: any, priority: number, domain: string): void {
    this.buffer.push({
      experience,
      priority: Math.pow(priority + 0.01, this.alpha),
      domain,
      timestamp: Date.now(),
    });

    if (this.buffer.length > this.maxSize) {
      this.buffer.sort((a, b) => b.priority - a.priority);
      this.buffer = this.buffer.slice(0, Math.floor(this.maxSize * 0.8));
    }
  }

  sample(batchSize: number, domain?: string): any[] {
    let candidates = this.buffer;
    if (domain) {
      candidates = this.buffer.filter(b => b.domain === domain);
    }

    if (candidates.length === 0) return [];

    const totalPriority = candidates.reduce((sum, b) => sum + b.priority, 0);
    const samples: any[] = [];

    for (let i = 0; i < Math.min(batchSize, candidates.length); i++) {
      let target = Math.random() * totalPriority;
      let cumSum = 0;

      for (const item of candidates) {
        cumSum += item.priority;
        if (cumSum >= target) {
          samples.push(item.experience);
          break;
        }
      }
    }

    return samples;
  }

  updatePriority(experience: any, newPriority: number): void {
    const item = this.buffer.find(b => b.experience === experience);
    if (item) {
      item.priority = Math.pow(newPriority + 0.01, this.alpha);
    }
  }

  size(): number {
    return this.buffer.length;
  }

  getStats(): Record<string, unknown> {
    const domains = [...new Set(this.buffer.map(b => b.domain))];
    return {
      totalSize: this.buffer.length,
      domains,
      avgPriority: this.buffer.reduce((s, b) => s + b.priority, 0) / (this.buffer.length || 1),
    };
  }
}

// ==================== LIFELONG LEARNER ====================

export class LifelongLearner {
  ewc: ElasticWeightConsolidation;
  pnn: ProgressiveNeuralNetwork;
  transfer: DomainTransferManager;
  curriculum: CurriculumManager;
  replay: PrioritizedExperienceReplay;

  private progressLog: LearningProgress[] = [];
  private currentDomain: string | null = null;

  constructor(config: {
    inputDim?: number;
    hiddenDim?: number;
    outputDim?: number;
    ewcLambda?: number;
    replaySize?: number;
  } = {}) {
    const { inputDim = 64, hiddenDim = 128, outputDim = 10, ewcLambda = 400, replaySize = 50000 } = config;

    this.ewc = new ElasticWeightConsolidation(ewcLambda);
    this.pnn = new ProgressiveNeuralNetwork(inputDim, hiddenDim, outputDim);
    this.transfer = new DomainTransferManager(inputDim, 32);
    this.curriculum = new CurriculumManager();
    this.replay = new PrioritizedExperienceReplay(replaySize);
  }

  async startDomain(domainName: string, sampleData?: number[][]): Promise<string> {
    const domainId = this.transfer.registerDomain(domainName, sampleData);
    this.pnn.addColumn(domainName);
    this.currentDomain = domainId;

    const bestSource = this.transfer.findBestTransferSource(domainId);
    if (bestSource) {
      await this.transfer.transfer(bestSource, domainId);
    }

    return domainId;
  }

  async learn(
    domainId: string,
    data: number[][],
    labels: number[][],
    epochs = 10
  ): Promise<{ loss: number; accuracy: number }> {
    const domain = this.transfer.getDomainKnowledge(domainId);
    if (!domain) return { loss: 1, accuracy: 0 };

    const column = this.pnn.getColumn(domain.name);
    if (!column) return { loss: 1, accuracy: 0 };

    const xs = tf.tensor2d(data);
    const ys = tf.tensor2d(labels);

    let totalLoss = 0;
    let totalAcc = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const result = await column.fit(xs, ys, {
        epochs: 1,
        verbose: 0,
        validationSplit: 0.1,
      });

      const loss = result.history.loss[0] as number;
      const ewcPenalty = this.ewc.computePenalty(column);
      const adjustedLoss = loss + ewcPenalty;

      totalLoss += adjustedLoss;
      totalAcc += (result.history.acc?.[0] as number) || 0;

      this.progressLog.push({
        domain: domain.name,
        taskId: `task_${Date.now()}`,
        epoch,
        loss: adjustedLoss,
        accuracy: (result.history.acc?.[0] as number) || 0,
        timestamp: Date.now(),
      });

      for (let i = 0; i < data.length; i++) {
        this.replay.add({ data: data[i], label: labels[i] }, 1 - loss, domain.name);
      }
    }

    await this.ewc.computeFisher(column, xs, ys, domainId);

    xs.dispose();
    ys.dispose();

    const avgLoss = totalLoss / epochs;
    const avgAcc = totalAcc / epochs;

    this.transfer.updateCompetence(domainId, avgAcc > 0.7, avgAcc);

    return { loss: avgLoss, accuracy: avgAcc };
  }

  async replay_train(domainId: string, batchSize = 32): Promise<{ loss: number }> {
    const domain = this.transfer.getDomainKnowledge(domainId);
    if (!domain) return { loss: 0 };

    const samples = this.replay.sample(batchSize, domain.name);
    if (samples.length === 0) return { loss: 0 };

    const data = samples.map(s => s.data);
    const labels = samples.map(s => s.label);

    const result = await this.learn(domainId, data, labels, 1);
    return { loss: result.loss };
  }

  getProgress(): {
    domains: DomainKnowledge[];
    curriculum: { completed: number; total: number; currentDifficulty: number };
    recentProgress: LearningProgress[];
  } {
    return {
      domains: this.transfer.getAllDomains(),
      curriculum: this.curriculum.getProgress(),
      recentProgress: this.progressLog.slice(-100),
    };
  }

  getCompetenceMap(): Map<string, number> {
    const map = new Map<string, number>();
    for (const domain of this.transfer.getAllDomains()) {
      map.set(domain.name, domain.competence);
    }
    return map;
  }

  exportState(): string {
    return JSON.stringify({
      domains: this.transfer.getAllDomains(),
      progress: this.curriculum.getProgress(),
      replayStats: this.replay.getStats(),
      progressLog: this.progressLog.slice(-1000),
    });
  }
}
