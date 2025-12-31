import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";
import fs from "fs";

/* =========================
   AM ORGANISM CORE
========================= */

export class AM_Organism {
  stateDim: number;
  actionDim: number;
  encoder: tf.LayersModel;
  worldModel: tf.LayersModel;

  constructor(stateDim: number, actionDim: number) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;

    const s = tf.input({ shape: [stateDim] });
    const h = tf.layers.dense({ units: 32, activation: "relu" }).apply(s) as tf.SymbolicTensor;
    const z = tf.layers.dense({ units: 8 }).apply(h) as tf.SymbolicTensor;
    this.encoder = tf.model({ inputs: s, outputs: z });

    const za = tf.input({ shape: [8 + actionDim] });
    const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(za) as tf.SymbolicTensor;
    const zp = tf.layers.dense({ units: 8 }).apply(h2) as tf.SymbolicTensor;
    this.worldModel = tf.model({ inputs: za, outputs: zp });
  }

  encode(state: number[]) {
    return tf.tidy(() =>
      Array.from((this.encoder.predict(tf.tensor2d([state])) as tf.Tensor).dataSync())
    );
  }

  predict(z: number[], a: number[]) {
    return tf.tidy(() =>
      Array.from(
        (this.worldModel.predict(tf.tensor2d([[...z, ...a]])) as tf.Tensor).dataSync()
      )
    );
  }
}

/* =========================
   OPTION POLICY (Skill Head)
========================= */

class OptionPolicy {
  model: tf.LayersModel;

  constructor(latentDim: number, actionDim: number) {
    const input = tf.input({ shape: [latentDim] });
    const h = tf.layers.dense({ units: 16, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const out = tf.layers
      .dense({ units: actionDim, activation: "tanh" })
      .apply(h) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: out });
    this.model.compile({ optimizer: "adam", loss: "meanSquaredError" });
  }

  async act(latent: number[]): Promise<number[]> {
    const t = tf.tensor2d([latent]);
    const pred = this.model.predict(t) as tf.Tensor;
    return Array.from(await pred.data());
  }

  async train(samples: { latent: number[]; action: number[] }[]) {
    if (!samples.length) return;
    const xs = tf.tensor2d(samples.map((s) => s.latent));
    const ys = tf.tensor2d(samples.map((s) => s.action));
    await this.model.fit(xs, ys, { epochs: 8, verbose: 0 });
  }
}

/* =========================
   AUTOENCODER (Latent)
========================= */

class AbstractionAutoencoder {
  model: tf.LayersModel;
  encoder: tf.LayersModel;
  latentDim: number;

  constructor(inputDim: number, latentDim = 8) {
    this.latentDim = latentDim;
    const input = tf.input({ shape: [inputDim] });
    const e1 = tf.layers.dense({ units: 16, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const latent = tf.layers
      .dense({ units: latentDim, activation: "sigmoid" })
      .apply(e1) as tf.SymbolicTensor;
    const d1 = tf.layers.dense({ units: 16, activation: "relu" }).apply(latent) as tf.SymbolicTensor;
    const out = tf.layers.dense({ units: inputDim }).apply(d1) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: out });
    this.encoder = tf.model({ inputs: input, outputs: latent });
    this.model.compile({ optimizer: "adam", loss: "meanSquaredError" });
  }

  async encode(state: number[]): Promise<number[]> {
    const t = tf.tensor2d([state]);
    const z = this.encoder.predict(t) as tf.Tensor;
    return Array.from(await z.data());
  }

  async train(states: number[][]) {
    if (!states.length) return;
    const t = tf.tensor2d(states);
    await this.model.fit(t, t, { epochs: 8, verbose: 0 });
  }
}

/* =========================
   WORLD MODEL
========================= */

class ConceptWorldModel {
  transitionModel: tf.LayersModel;
  latentDim: number;
  actionDim: number;

  constructor(latentDim: number, actionDim: number) {
    this.latentDim = latentDim;
    this.actionDim = actionDim;

    const input = tf.input({ shape: [latentDim + actionDim] });
    const h = tf.layers.dense({ units: 32, activation: "relu" }).apply(input) as tf.SymbolicTensor;
    const out = tf.layers.dense({ units: latentDim }).apply(h) as tf.SymbolicTensor;

    this.transitionModel = tf.model({ inputs: input, outputs: out });
    this.transitionModel.compile({ optimizer: "adam", loss: "meanSquaredError" });
  }

  async predict(latent: number[], action: number[]): Promise<number[]> {
    const t = tf.tensor2d([[...latent, ...action]]);
    const z = this.transitionModel.predict(t) as tf.Tensor;
    return Array.from(await z.data());
  }

  async train(pairs: { z: number[]; a: number[]; zNext: number[] }[]) {
    if (!pairs.length) return;
    const xs = tf.tensor2d(pairs.map((p) => [...p.z, ...p.a]));
    const ys = tf.tensor2d(pairs.map((p) => p.zNext));
    await this.transitionModel.fit(xs, ys, { epochs: 8, verbose: 0 });
  }
}

/* =========================
   SYMBOLS
========================= */

class LatentSymbolExtractor {
  k: number;
  centroids: number[][] = [];
  symbolIds: string[] = [];
  constructor(k = 12) {
    this.k = k;
  }
  extract(latents: number[][]) {
    if (latents.length < this.k) return;
    this.centroids = [];
    this.symbolIds = [];
    for (let i = 0; i < this.k; i++) {
      const idx = Math.floor((i * latents.length) / this.k);
      this.centroids.push(latents[idx]);
      this.symbolIds.push("S" + i);
    }
  }
  assign(latent: number[]): string {
    if (!this.centroids.length) return "S0";
    let min = Infinity,
      best = 0;
    for (let i = 0; i < this.centroids.length; i++) {
      const d = math.norm(math.subtract(latent, this.centroids[i])) as unknown as number;
      if (d < min) {
        min = d;
        best = i;
      }
    }
    return this.symbolIds[best];
  }
}

/* =========================
   METRICS
========================= */

class SymbolPhaseMetrics {
  usages: Record<string, number> = {};
  inTasks: Record<string, Set<string>> = {};
  transitions: Record<string, Set<string>> = {};
  generalizes: Record<string, boolean> = {};
  reusable: Record<string, boolean> = {};
  epoch = 0;

  observe(sym: string, task: string, next: string) {
    this.usages[sym] = (this.usages[sym] || 0) + 1;
    this.inTasks[sym] ||= new Set();
    this.inTasks[sym].add(task);
    this.transitions[sym] ||= new Set();
    if (next) this.transitions[sym].add(next);

    if (this.inTasks[sym].size > 1 && this.usages[sym] > 10 && (this.transitions[sym]?.size || 0) > 3)
      this.generalizes[sym] = true;
    if (this.inTasks[sym].size > 2 && this.usages[sym] > 7) this.reusable[sym] = true;
  }

  nextEpoch() {
    this.epoch++;
  }

  snapshot() {
    return Object.keys(this.usages).map((s) => ({
      s,
      usage: this.usages[s],
      generalizes: !!this.generalizes[s],
      reusable: !!this.reusable[s],
      tasks: [...(this.inTasks[s] || [])],
      transitions: [...(this.transitions[s] || [])],
    }));
  }
}

/* =========================
   SYMBOL GRAPH
========================= */

class SymbolNode {
  id: string;
  vector: number[];
  parent: string | null = null;
  children = new Set<string>();
  constructor(id: string, vector: number[]) {
    this.id = id;
    this.vector = vector;
  }
}

class HierarchicalSymbolGraph {
  nodes: Record<string, SymbolNode> = {};
  programs: Record<string, string[]> = {};

  addSymbol(id: string, vector: number[]) {
    if (!this.nodes[id]) this.nodes[id] = new SymbolNode(id, vector);
  }

  addChild(p: string, c: string) {
    if (this.nodes[p] && this.nodes[c]) {
      this.nodes[p].children.add(c);
      this.nodes[c].parent = p;
    }
  }

  compose(meta: string, seq: string[]) {
    this.programs[meta] = seq;
  }

  hierarchy() {
    return Object.fromEntries(
      Object.entries(this.nodes).map(([k, n]) => [k, { parent: n.parent, children: [...n.children] }])
    );
  }
}

/* =========================
   PLANNER
========================= */

class CounterfactualPlanner {
  core: AMKernel;
  graph: HierarchicalSymbolGraph;

  constructor(core: AMKernel, graph: HierarchicalSymbolGraph) {
    this.core = core;
    this.graph = graph;
  }

  async score(latent: number[], metaId: string, steps = 5) {
    let curr = latent;
    let score = 0;
    const seq: any[] = [];

    for (let i = 0; i < steps; i++) {
      let action = Array.from({ length: this.core.actionDim }, () => Math.random());
      if (this.core.optionPolicies[metaId]) action = await this.core.optionPolicies[metaId].act(curr);

      const out = await this.core.worldModel.predict(curr, action);

      const worldLik = -1 * (math.norm(out) as unknown as number);
      const compression = -1 * (math.variance(out) as unknown as number);
      const utility = math.sum(out) as unknown as number;
      const stability = math.std(out) as unknown as number;

      const step = worldLik + 0.2 * compression + 0.5 * utility - 0.1 * stability;
      score += step;
      curr = out;
      seq.push({ out, step });
    }
    return { metaId, score, seq };
  }

  async plan(latent: number[], metas: string[]) {
    if (!metas.length) return null;
    const scored = await Promise.all(metas.map((m) => this.score(latent, m, 6)));
    return scored.sort((a, b) => b.score - a.score)[0];
  }
}

/* =========================
   KERNEL
========================= */

type EnvObs = { state: number[]; reward: number; done: boolean };
type Env = { reset: () => Promise<EnvObs>; step: (a: number[]) => Promise<EnvObs> };

class AMKernel {
  stateDim: number;
  actionDim: number;
  env: Env;

  autoencoder: AbstractionAutoencoder;
  worldModel: ConceptWorldModel;

  symboler = new LatentSymbolExtractor(12);
  metrics = new SymbolPhaseMetrics();
  graph = new HierarchicalSymbolGraph();
  planner: CounterfactualPlanner;

  optionPolicies: Record<string, OptionPolicy> = {};
  optionTrainData: Record<string, { latent: number[]; action: number[] }[]> = {};
  replay: { s: number[]; a: number[]; z: number[]; zNext: number[]; sym: string; nextSym: string }[] =
    [];
  logs: any[] = [];

  constructor(env: Env, stateDim: number, actionDim: number) {
    this.env = env;
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.autoencoder = new AbstractionAutoencoder(stateDim, 8);
    this.worldModel = new ConceptWorldModel(8, actionDim);
    this.planner = new CounterfactualPlanner(this, this.graph);
  }

  async runEpisode(taskId = "task0") {
    let obs = await this.env.reset();
    const latents: number[][] = [];
    const symbols: string[] = [];

    for (let t = 0; t < 50; t++) {
      const z = await this.autoencoder.encode(obs.state);
      latents.push(z);

      const sym = this.symboler.assign(z);
      const prev = symbols.length ? symbols[symbols.length - 1] : "";
      this.metrics.observe(sym, taskId, prev);
      this.graph.addSymbol(sym, z);
      if (prev) this.graph.addChild(prev, sym);
      symbols.push(sym);

      for (const metaId of Object.keys(this.graph.programs)) {
        if (!this.optionTrainData[metaId]) this.optionTrainData[metaId] = [];
        if (this.graph.programs[metaId].includes(sym)) {
          const action = Array.from({ length: this.actionDim }, () => Math.random());
          this.optionTrainData[metaId].push({ latent: z, action });
        }
      }

      const action = Array.from({ length: this.actionDim }, () => Math.random());
      const nextObs = await this.env.step(action);
      const zNext = await this.autoencoder.encode(nextObs.state);
      this.replay.push({ s: obs.state, a: action, z, zNext, sym, nextSym: "" });

      obs = nextObs;
      if (obs.done) break;
    }

    this.symboler.extract(latents);
    await this.worldModel.train(this.replay.map((r) => ({ z: r.z, a: r.a, zNext: r.zNext })));
    await this.autoencoder.train(this.replay.map((r) => r.s));
    this.metrics.nextEpoch();

    for (const s of Object.keys(this.metrics.usages)) {
      if (this.metrics.generalizes[s] && this.metrics.reusable[s]) {
        const metaId = "META_" + s;
        if (!this.graph.programs[metaId]) {
          this.graph.compose(metaId, [s]);
          this.optionPolicies[metaId] = new OptionPolicy(8, this.actionDim);
        }
        const trainData = this.optionTrainData[metaId] || [];
        if (trainData.length > 10) {
          await this.optionPolicies[metaId].train(trainData.splice(0, trainData.length));
        }
      }
    }

    const metas = Object.keys(this.graph.programs);
    if (metas.length) {
      const best = await this.planner.plan(latents[0], metas);
      this.logs.push({ metrics: this.metrics.snapshot(), best });
      fs.writeFileSync("am-kernel-metrics.json", JSON.stringify(this.logs.slice(-10), null, 2));
    }
  }
}

/* =========================
   DUMMY ENV
========================= */

class DummyEnv implements Env {
  async reset() {
    return { state: [0, 0, 0, 0], reward: 0, done: false };
  }
  async step(_: number[]) {
    const s = [Math.random(), Math.random(), Math.random(), Math.random()];
    return { state: s, reward: (math.sum(s) as unknown as number), done: Math.random() > 0.95 };
  }
}

export const core = new AMKernel(new DummyEnv(), 4, 2);
export { AMKernel };

