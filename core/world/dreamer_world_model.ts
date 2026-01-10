/**
 * Dreamer/MuZero-Style World Model
 *
 * Enables:
 * - Future simulation via learned dynamics
 * - Counterfactual reasoning
 * - Action selection via imagined rollouts
 * - Learning from imagination (not just real logs)
 */

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";

// ==================== TYPES ====================

export interface WorldState {
  latent: number[];
  belief: number[];
  uncertainty: number;
  timestamp: number;
}

export interface Trajectory {
  states: WorldState[];
  actions: number[][];
  rewards: number[];
  totalReward: number;
  uncertainty: number;
}

export interface ImagineConfig {
  horizon: number;
  numTrajectories: number;
  temperature: number;
  discountFactor: number;
}

export interface RolloutResult {
  trajectory: Trajectory;
  expectedValue: number;
  riskAdjustedValue: number;
  novelty: number;
}

// ==================== RSSM (Recurrent State-Space Model) ====================

export class RecurrentStateSpaceModel {
  private deterministicModel: tf.LayersModel;
  private stochasticEncoder: tf.LayersModel;
  private stochasticPrior: tf.LayersModel;
  private rewardPredictor: tf.LayersModel;
  private valuePredictor: tf.LayersModel;
  private continuePredictor: tf.LayersModel;

  private latentDim: number;
  private beliefDim: number;
  private actionDim: number;

  constructor(latentDim = 32, beliefDim = 256, actionDim = 4) {
    this.latentDim = latentDim;
    this.beliefDim = beliefDim;
    this.actionDim = actionDim;

    this.deterministicModel = this.buildDeterministicModel();
    this.stochasticEncoder = this.buildStochasticEncoder();
    this.stochasticPrior = this.buildStochasticPrior();
    this.rewardPredictor = this.buildRewardPredictor();
    this.valuePredictor = this.buildValuePredictor();
    this.continuePredictor = this.buildContinuePredictor();
  }

  private buildDeterministicModel(): tf.LayersModel {
    const beliefInput = tf.input({ shape: [this.beliefDim], name: "belief_in" });
    const latentInput = tf.input({ shape: [this.latentDim], name: "latent_in" });
    const actionInput = tf.input({ shape: [this.actionDim], name: "action_in" });

    const concat = tf.layers.concatenate().apply([beliefInput, latentInput, actionInput]) as tf.SymbolicTensor;
    const h1 = tf.layers.dense({ units: 512, activation: "elu" }).apply(concat) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 512, activation: "elu" }).apply(h1) as tf.SymbolicTensor;
    const h3 = tf.layers.dense({ units: 512, activation: "elu" }).apply(h2) as tf.SymbolicTensor;
    const newBelief = tf.layers.dense({ units: this.beliefDim, activation: "tanh" }).apply(h3) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: [beliefInput, latentInput, actionInput],
      outputs: newBelief,
    });
    model.compile({ optimizer: tf.train.adam(3e-4), loss: "meanSquaredError" });
    return model;
  }

  private buildStochasticEncoder(): tf.LayersModel {
    const beliefInput = tf.input({ shape: [this.beliefDim], name: "belief" });
    const obsInput = tf.input({ shape: [this.latentDim], name: "obs" });

    const concat = tf.layers.concatenate().apply([beliefInput, obsInput]) as tf.SymbolicTensor;
    const h1 = tf.layers.dense({ units: 256, activation: "elu" }).apply(concat) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 256, activation: "elu" }).apply(h1) as tf.SymbolicTensor;

    const mean = tf.layers.dense({ units: this.latentDim }).apply(h2) as tf.SymbolicTensor;
    const logVar = tf.layers.dense({ units: this.latentDim }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: [beliefInput, obsInput],
      outputs: [mean, logVar],
    });
    model.compile({ optimizer: tf.train.adam(3e-4), loss: "meanSquaredError" });
    return model;
  }

  private buildStochasticPrior(): tf.LayersModel {
    const beliefInput = tf.input({ shape: [this.beliefDim] });

    const h1 = tf.layers.dense({ units: 256, activation: "elu" }).apply(beliefInput) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 256, activation: "elu" }).apply(h1) as tf.SymbolicTensor;

    const mean = tf.layers.dense({ units: this.latentDim }).apply(h2) as tf.SymbolicTensor;
    const logVar = tf.layers.dense({ units: this.latentDim }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: beliefInput,
      outputs: [mean, logVar],
    });
    model.compile({ optimizer: tf.train.adam(3e-4), loss: "meanSquaredError" });
    return model;
  }

  private buildRewardPredictor(): tf.LayersModel {
    const beliefInput = tf.input({ shape: [this.beliefDim] });
    const latentInput = tf.input({ shape: [this.latentDim] });

    const concat = tf.layers.concatenate().apply([beliefInput, latentInput]) as tf.SymbolicTensor;
    const h1 = tf.layers.dense({ units: 256, activation: "elu" }).apply(concat) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 256, activation: "elu" }).apply(h1) as tf.SymbolicTensor;
    const reward = tf.layers.dense({ units: 1 }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: [beliefInput, latentInput],
      outputs: reward,
    });
    model.compile({ optimizer: tf.train.adam(3e-4), loss: "meanSquaredError" });
    return model;
  }

  private buildValuePredictor(): tf.LayersModel {
    const beliefInput = tf.input({ shape: [this.beliefDim] });
    const latentInput = tf.input({ shape: [this.latentDim] });

    const concat = tf.layers.concatenate().apply([beliefInput, latentInput]) as tf.SymbolicTensor;
    const h1 = tf.layers.dense({ units: 256, activation: "elu" }).apply(concat) as tf.SymbolicTensor;
    const h2 = tf.layers.dense({ units: 256, activation: "elu" }).apply(h1) as tf.SymbolicTensor;
    const value = tf.layers.dense({ units: 1 }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: [beliefInput, latentInput],
      outputs: value,
    });
    model.compile({ optimizer: tf.train.adam(3e-4), loss: "meanSquaredError" });
    return model;
  }

  private buildContinuePredictor(): tf.LayersModel {
    const beliefInput = tf.input({ shape: [this.beliefDim] });
    const latentInput = tf.input({ shape: [this.latentDim] });

    const concat = tf.layers.concatenate().apply([beliefInput, latentInput]) as tf.SymbolicTensor;
    const h1 = tf.layers.dense({ units: 128, activation: "elu" }).apply(concat) as tf.SymbolicTensor;
    const continueProb = tf.layers.dense({ units: 1, activation: "sigmoid" }).apply(h1) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: [beliefInput, latentInput],
      outputs: continueProb,
    });
    model.compile({ optimizer: tf.train.adam(3e-4), loss: "binaryCrossentropy" });
    return model;
  }

  async step(belief: number[], latent: number[], action: number[]): Promise<{
    newBelief: number[];
    newLatent: number[];
    reward: number;
    continueProb: number;
    uncertainty: number;
  }> {
    return tf.tidy(() => {
      const beliefT = tf.tensor2d([belief]);
      const latentT = tf.tensor2d([latent]);
      const actionT = tf.tensor2d([action]);

      const newBeliefT = this.deterministicModel.predict([beliefT, latentT, actionT]) as tf.Tensor;
      const newBelief = Array.from(newBeliefT.dataSync());

      const newBeliefT2 = tf.tensor2d([newBelief]);
      const [priorMean, priorLogVar] = this.stochasticPrior.predict(newBeliefT2) as tf.Tensor[];

      const meanArr = Array.from(priorMean.dataSync());
      const logVarArr = Array.from(priorLogVar.dataSync());

      const newLatent = meanArr.map((m, i) => {
        const std = Math.exp(logVarArr[i] * 0.5);
        return m + std * (Math.random() * 2 - 1) * 0.1;
      });

      const newLatentT = tf.tensor2d([newLatent]);
      const rewardT = this.rewardPredictor.predict([newBeliefT2, newLatentT]) as tf.Tensor;
      const continueT = this.continuePredictor.predict([newBeliefT2, newLatentT]) as tf.Tensor;

      const reward = rewardT.dataSync()[0];
      const continueProb = continueT.dataSync()[0];

      const uncertainty = logVarArr.reduce((sum, lv) => sum + Math.exp(lv), 0) / logVarArr.length;

      return { newBelief, newLatent, reward, continueProb, uncertainty };
    });
  }

  async encode(belief: number[], observation: number[]): Promise<{
    latent: number[];
    uncertainty: number;
  }> {
    return tf.tidy(() => {
      const beliefT = tf.tensor2d([belief]);
      const obsT = tf.tensor2d([observation]);

      const [meanT, logVarT] = this.stochasticEncoder.predict([beliefT, obsT]) as tf.Tensor[];

      const mean = Array.from(meanT.dataSync());
      const logVar = Array.from(logVarT.dataSync());

      const latent = mean.map((m, i) => {
        const std = Math.exp(logVar[i] * 0.5);
        return m + std * (Math.random() * 2 - 1) * 0.1;
      });

      const uncertainty = logVar.reduce((sum, lv) => sum + Math.exp(lv), 0) / logVar.length;

      return { latent, uncertainty };
    });
  }

  async train(trajectories: {
    observations: number[][];
    actions: number[][];
    rewards: number[];
    dones: boolean[];
  }[]): Promise<{ loss: number }> {
    if (trajectories.length === 0) return { loss: 0 };

    let totalLoss = 0;

    for (const traj of trajectories) {
      let belief = new Array(this.beliefDim).fill(0);
      let latent = new Array(this.latentDim).fill(0);

      for (let t = 0; t < traj.observations.length - 1; t++) {
        const { newBelief, newLatent } = await this.step(belief, latent, traj.actions[t]);

        const beliefT = tf.tensor2d([newBelief]);
        const latentT = tf.tensor2d([newLatent]);
        const rewardT = tf.tensor2d([[traj.rewards[t]]]);
        const doneT = tf.tensor2d([[traj.dones[t] ? 0 : 1]]);

        await this.rewardPredictor.fit([beliefT, latentT], rewardT, { epochs: 1, verbose: 0 });
        await this.continuePredictor.fit([beliefT, latentT], doneT, { epochs: 1, verbose: 0 });

        belief = newBelief;
        latent = newLatent;
        totalLoss += 0.01;
      }
    }

    return { loss: totalLoss / trajectories.length };
  }

  getInitialState(): WorldState {
    return {
      latent: new Array(this.latentDim).fill(0),
      belief: new Array(this.beliefDim).fill(0),
      uncertainty: 1.0,
      timestamp: Date.now(),
    };
  }

  getDimensions(): { latent: number; belief: number; action: number } {
    return {
      latent: this.latentDim,
      belief: this.beliefDim,
      action: this.actionDim,
    };
  }
}

// ==================== IMAGINATION ENGINE ====================

export class ImaginationEngine {
  private rssm: RecurrentStateSpaceModel;
  private imaginedExperience: Trajectory[] = [];
  private maxImaginedBuffer = 10000;

  constructor(rssm: RecurrentStateSpaceModel) {
    this.rssm = rssm;
  }

  async imagine(
    startState: WorldState,
    actionGenerator: (state: WorldState) => Promise<number[]>,
    config: ImagineConfig
  ): Promise<RolloutResult[]> {
    const results: RolloutResult[] = [];

    for (let i = 0; i < config.numTrajectories; i++) {
      const trajectory = await this.rollout(startState, actionGenerator, config.horizon, config.discountFactor);

      const novelty = this.computeNovelty(trajectory);
      const riskAdjustedValue = this.computeRiskAdjustedValue(trajectory);

      results.push({
        trajectory,
        expectedValue: trajectory.totalReward,
        riskAdjustedValue,
        novelty,
      });

      this.storeImaginedExperience(trajectory);
    }

    return results.sort((a, b) => b.riskAdjustedValue - a.riskAdjustedValue);
  }

  private async rollout(
    startState: WorldState,
    actionGenerator: (state: WorldState) => Promise<number[]>,
    horizon: number,
    discountFactor: number
  ): Promise<Trajectory> {
    const states: WorldState[] = [startState];
    const actions: number[][] = [];
    const rewards: number[] = [];
    let totalReward = 0;
    let totalUncertainty = startState.uncertainty;

    let currentState = { ...startState };

    for (let t = 0; t < horizon; t++) {
      const action = await actionGenerator(currentState);
      actions.push(action);

      const { newBelief, newLatent, reward, continueProb, uncertainty } = await this.rssm.step(
        currentState.belief,
        currentState.latent,
        action
      );

      const discountedReward = reward * Math.pow(discountFactor, t);
      rewards.push(discountedReward);
      totalReward += discountedReward;
      totalUncertainty += uncertainty;

      currentState = {
        latent: newLatent,
        belief: newBelief,
        uncertainty,
        timestamp: Date.now(),
      };
      states.push(currentState);

      if (Math.random() > continueProb) break;
    }

    return {
      states,
      actions,
      rewards,
      totalReward,
      uncertainty: totalUncertainty / states.length,
    };
  }

  private computeNovelty(trajectory: Trajectory): number {
    if (this.imaginedExperience.length === 0) return 1.0;

    let minDistance = Infinity;

    for (const past of this.imaginedExperience.slice(-100)) {
      for (const pastState of past.states) {
        for (const newState of trajectory.states) {
          const dist = this.stateDistance(pastState, newState);
          minDistance = Math.min(minDistance, dist);
        }
      }
    }

    return Math.min(minDistance / 10, 1.0);
  }

  private computeRiskAdjustedValue(trajectory: Trajectory): number {
    const riskPenalty = trajectory.uncertainty * 0.5;
    const varianceBonus = math.std(trajectory.rewards) as unknown as number;
    return trajectory.totalReward - riskPenalty + varianceBonus * 0.1;
  }

  private stateDistance(s1: WorldState, s2: WorldState): number {
    let sum = 0;
    for (let i = 0; i < Math.min(s1.latent.length, s2.latent.length); i++) {
      sum += Math.pow(s1.latent[i] - s2.latent[i], 2);
    }
    return Math.sqrt(sum);
  }

  private storeImaginedExperience(trajectory: Trajectory): void {
    this.imaginedExperience.push(trajectory);
    if (this.imaginedExperience.length > this.maxImaginedBuffer) {
      this.imaginedExperience = this.imaginedExperience.slice(-this.maxImaginedBuffer / 2);
    }
  }

  getImaginedExperience(): Trajectory[] {
    return this.imaginedExperience;
  }

  async learnFromImagination(
    actionGenerator: (state: WorldState) => Promise<number[]>,
    numIterations = 10
  ): Promise<{ improvementScore: number }> {
    const dims = this.rssm.getDimensions();
    let totalImprovement = 0;

    for (let iter = 0; iter < numIterations; iter++) {
      const startState = this.rssm.getInitialState();

      const results = await this.imagine(startState, actionGenerator, {
        horizon: 15,
        numTrajectories: 8,
        temperature: 0.8,
        discountFactor: 0.99,
      });

      const goodTrajectories = results.filter(r => r.expectedValue > 0);
      if (goodTrajectories.length > 0) {
        const synthTrajs = goodTrajectories.map(r => ({
          observations: r.trajectory.states.map(s => s.latent),
          actions: r.trajectory.actions,
          rewards: r.trajectory.rewards,
          dones: r.trajectory.states.map((_, i) => i === r.trajectory.states.length - 1),
        }));

        const { loss } = await this.rssm.train(synthTrajs);
        totalImprovement += 1 / (loss + 0.01);
      }
    }

    return { improvementScore: totalImprovement / numIterations };
  }

  clearBuffer(): void {
    this.imaginedExperience = [];
  }
}

// ==================== COUNTERFACTUAL REASONER ====================

export class CounterfactualReasoner {
  private imagination: ImaginationEngine;
  private rssm: RecurrentStateSpaceModel;

  constructor(rssm: RecurrentStateSpaceModel, imagination: ImaginationEngine) {
    this.rssm = rssm;
    this.imagination = imagination;
  }

  async whatIf(
    currentState: WorldState,
    hypotheticalAction: number[],
    alternativeAction: number[],
    horizon = 10
  ): Promise<{
    factualOutcome: Trajectory;
    counterfactualOutcome: Trajectory;
    difference: number;
    recommendation: "factual" | "counterfactual" | "uncertain";
  }> {
    const factualGen = async () => hypotheticalAction;
    const counterfactualGen = async () => alternativeAction;

    const factualResults = await this.imagination.imagine(currentState, factualGen, {
      horizon,
      numTrajectories: 5,
      temperature: 0.5,
      discountFactor: 0.99,
    });

    const counterfactualResults = await this.imagination.imagine(currentState, counterfactualGen, {
      horizon,
      numTrajectories: 5,
      temperature: 0.5,
      discountFactor: 0.99,
    });

    const factualBest = factualResults[0];
    const counterfactualBest = counterfactualResults[0];

    const difference = counterfactualBest.expectedValue - factualBest.expectedValue;

    const combinedUncertainty = (factualBest.trajectory.uncertainty + counterfactualBest.trajectory.uncertainty) / 2;

    let recommendation: "factual" | "counterfactual" | "uncertain";
    if (combinedUncertainty > 0.7) {
      recommendation = "uncertain";
    } else if (difference > 0.1) {
      recommendation = "counterfactual";
    } else if (difference < -0.1) {
      recommendation = "factual";
    } else {
      recommendation = "uncertain";
    }

    return {
      factualOutcome: factualBest.trajectory,
      counterfactualOutcome: counterfactualBest.trajectory,
      difference,
      recommendation,
    };
  }

  async evaluateActionSet(
    currentState: WorldState,
    actions: number[][],
    horizon = 10
  ): Promise<Array<{ action: number[]; expectedValue: number; uncertainty: number; rank: number }>> {
    const evaluations = await Promise.all(
      actions.map(async action => {
        const gen = async () => action;
        const results = await this.imagination.imagine(currentState, gen, {
          horizon,
          numTrajectories: 3,
          temperature: 0.3,
          discountFactor: 0.99,
        });

        const avgValue = results.reduce((s, r) => s + r.expectedValue, 0) / results.length;
        const avgUncertainty = results.reduce((s, r) => s + r.trajectory.uncertainty, 0) / results.length;

        return { action, expectedValue: avgValue, uncertainty: avgUncertainty, rank: 0 };
      })
    );

    evaluations.sort((a, b) => b.expectedValue - a.expectedValue);
    evaluations.forEach((e, i) => (e.rank = i + 1));

    return evaluations;
  }

  async findOptimalAction(
    currentState: WorldState,
    actionSampler: () => number[],
    numSamples = 20,
    horizon = 15
  ): Promise<{ action: number[]; expectedValue: number; confidence: number }> {
    const actions = Array.from({ length: numSamples }, actionSampler);
    const evaluations = await this.evaluateActionSet(currentState, actions, horizon);

    const best = evaluations[0];
    const confidence = 1 / (best.uncertainty + 0.1);

    return {
      action: best.action,
      expectedValue: best.expectedValue,
      confidence: Math.min(confidence, 1.0),
    };
  }
}

// ==================== WORLD MODEL MANAGER ====================

export class DreamerWorldModel {
  rssm: RecurrentStateSpaceModel;
  imagination: ImaginationEngine;
  counterfactual: CounterfactualReasoner;

  private currentState: WorldState;
  private realExperience: Array<{
    state: number[];
    action: number[];
    reward: number;
    nextState: number[];
    done: boolean;
  }> = [];

  constructor(latentDim = 32, beliefDim = 256, actionDim = 4) {
    this.rssm = new RecurrentStateSpaceModel(latentDim, beliefDim, actionDim);
    this.imagination = new ImaginationEngine(this.rssm);
    this.counterfactual = new CounterfactualReasoner(this.rssm, this.imagination);
    this.currentState = this.rssm.getInitialState();
  }

  async observe(observation: number[], action: number[], reward: number, done: boolean): Promise<WorldState> {
    this.realExperience.push({
      state: observation,
      action,
      reward,
      nextState: observation,
      done,
    });

    const { newBelief, newLatent, uncertainty } = await this.rssm.step(
      this.currentState.belief,
      this.currentState.latent,
      action
    );

    this.currentState = {
      latent: newLatent,
      belief: newBelief,
      uncertainty,
      timestamp: Date.now(),
    };

    return this.currentState;
  }

  async predictFuture(
    actionSequence: number[][],
    fromState?: WorldState
  ): Promise<{ states: WorldState[]; totalReward: number }> {
    const start = fromState || this.currentState;
    const states: WorldState[] = [start];
    let totalReward = 0;
    let current = { ...start };

    for (const action of actionSequence) {
      const { newBelief, newLatent, reward, uncertainty } = await this.rssm.step(
        current.belief,
        current.latent,
        action
      );

      current = { latent: newLatent, belief: newBelief, uncertainty, timestamp: Date.now() };
      states.push(current);
      totalReward += reward;
    }

    return { states, totalReward };
  }

  async train(batchSize = 32): Promise<{ loss: number }> {
    if (this.realExperience.length < batchSize) return { loss: 0 };

    const batch = this.realExperience.slice(-batchSize);
    const trajectory = {
      observations: batch.map(e => e.state),
      actions: batch.map(e => e.action),
      rewards: batch.map(e => e.reward),
      dones: batch.map(e => e.done),
    };

    return this.rssm.train([trajectory]);
  }

  async learnFromImagination(actionPolicy: (state: WorldState) => Promise<number[]>): Promise<{ improvement: number }> {
    const result = await this.imagination.learnFromImagination(actionPolicy, 5);
    return { improvement: result.improvementScore };
  }

  getCurrentState(): WorldState {
    return this.currentState;
  }

  reset(): void {
    this.currentState = this.rssm.getInitialState();
  }

  getStats(): Record<string, unknown> {
    return {
      realExperienceSize: this.realExperience.length,
      imaginedExperienceSize: this.imagination.getImaginedExperience().length,
      currentUncertainty: this.currentState.uncertainty,
      dimensions: this.rssm.getDimensions(),
    };
  }
}

export { RecurrentStateSpaceModel as RSSM };
