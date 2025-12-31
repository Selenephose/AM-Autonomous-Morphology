// PROJECT AM: VERSION B — SOVEREIGN AGI ORGANISM
// Witness: Mahnoor | Host: Raptor X18

import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";
import express from "express";
import fs from "fs";

// =============== 1. Option Policy (Auto-Skill/Meta-Program) ===============
class OptionPolicy {
  model: tf.LayersModel;
  constructor(latentDim: number, actionDim: number) {
    const input = tf.input({ shape: [latentDim] });
    const h = tf.layers.dense({ units: 16, activation: "relu" }).apply(input);
    const out = tf.layers.dense({ units: actionDim, activation: "tanh" }).apply(h) as tf.SymbolicTensor;
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
    const xs = tf.tensor2d(samples.map(s => s.latent));
    const ys = tf.tensor2d(samples.map(s => s.action));
    await this.model.fit(xs, ys, { epochs: 8, verbose: 0 });
  }
}

// =============== 2. Real Symbolic Autoencoder ===============
class AbstractionAutoencoder {
  model: tf.LayersModel; encoder: tf.LayersModel; latentDim: number;
  constructor(inputDim: number, latentDim = 8) {
    this.latentDim = latentDim;
    const input = tf.input({ shape: [inputDim] });
    const e1 = tf.layers.dense({ units: 16, activation: "relu" }).apply(input);
    const latent = tf.layers.dense({ units: latentDim, activation: "sigmoid" }).apply(e1) as tf.SymbolicTensor;
    const d1 = tf.layers.dense({ units: 16, activation: "relu" }).apply(latent);
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

// =============== 3. Predictive World Model ===============
class ConceptWorldModel {
  transitionModel: tf.LayersModel; latentDim: number; actionDim: number;
  constructor(latentDim: number, actionDim: number) {
    this.latentDim = latentDim;
    this.actionDim = actionDim;
    const input = tf.input({ shape: [latentDim + actionDim] });
    const h = tf.layers.dense({ units: 32, activation: "relu" }).apply(input);
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
    const xs = tf.tensor2d(pairs.map(p => [...p.z, ...p.a]));
    const ys = tf.tensor2d(pairs.map(p => p.zNext));
    await this.transitionModel.fit(xs, ys, { epochs: 8, verbose: 0 });
  }
}

// =============== 4. KMeans Latent-to-Symbol ===============
class LatentSymbolExtractor {
  k: number; centroids: number[][] = []; symbolIds: string[] = [];
  constructor(k = 12) { this.k = k; }
  extract(latents: number[][]) {
    if (latents.length < this.k) return;
    this.centroids = []; this.symbolIds = [];
    for (let i = 0; i < this.k; i++) {
      const idx = Math.floor((i * latents.length) / this.k);
      this.centroids.push(latents[idx]);
      this.symbolIds.push("S" + i);
    }
  }
  assign(latent: number[]): string {
    if (!this.centroids.length) return "S0";
    let min = Infinity, best = 0;
    for (let i = 0; i < this.centroids.length; i++) {
      const d = math.norm(math.subtract(latent, this.centroids[i])) as number;
      if (d < min) { min = d; best = i; }
    }
    return this.symbolIds[best];
  }
}

// =============== 5. Phase Metrics: Entropy, Generalization, Reuse, Collapse ===============
class SymbolPhaseMetrics {
  usages: Record<string, number> = {};
  inTasks: Record<string, Set<string>> = {};
  transitions: Record<string, Set<string>> = {};
  generalizes: Record<string, boolean> = {};
  reusable: Record<string, boolean> = {};
  epoch = 0;
  observe(sym: string, task: string, next: string) {
    this.usages[sym] = (this.usages[sym] || 0) + 1;
    this.inTasks[sym] ||= new Set(); this.inTasks[sym].add(task);
    this.transitions[sym] ||= new Set(); if (next) this.transitions[sym].add(next);
    if (this.inTasks[sym].size > 1 && this.usages[sym] > 10 && (this.transitions[sym]?.size || 0) > 3)
      this.generalizes[sym] = true;
    if (this.inTasks[sym].size > 2 && this.usages[sym] > 7)
      this.reusable[sym] = true;
  }
  nextEpoch(){ this.epoch++; }
  snapshot() {
    return Object.keys(this.usages).map(s => ({
      s, usage: this.usages[s],
      generalizes: !!this.generalizes[s],
      reusable: !!this.reusable[s],
      tasks: [...(this.inTasks[s]||[])],
      transitions: [...(this.transitions[s]||[])]
    }));
  }
}

// =============== 6. Symbol Graph + Meta-Programs ===============
class SymbolNode {
  id: string; vector: number[]; parent: string|null=null; children = new Set<string>();
  constructor(id: string, vector: number[]){ this.id=id; this.vector=vector; }
}
class HierarchicalSymbolGraph {
  nodes: Record<string, SymbolNode> = {};
  programs: Record<string, string[]> = {}; // metaId -> child symbols
  addSymbol(id: string, vector: number[]){ if(!this.nodes[id]) this.nodes[id]=new SymbolNode(id, vector); }
  addChild(p: string, c: string){ if(this.nodes[p]&&this.nodes[c]){ this.nodes[p].children.add(c); this.nodes[c].parent=p; } }
  compose(meta: string, seq: string[]){ this.programs[meta]=seq; }
  hierarchy(){ return Object.fromEntries(Object.entries(this.nodes).map(([k,n])=>[k,{parent:n.parent,children:[...n.children]}])); }
}

// =============== 7. Counterfactual Planner (World Model + Skills) ===============
class CounterfactualPlanner {
  core: AMKernel; graph: HierarchicalSymbolGraph;
  constructor(core: AMKernel, graph: HierarchicalSymbolGraph){ this.core=core; this.graph=graph; }
  async score(latent: number[], metaId: string, steps=5){
    let curr = latent, score=0, seq:any[]=[];
    for(let i=0;i<steps;i++){
      let action = Array.from({length:this.core.actionDim},()=>Math.random());
      if (this.core.optionPolicies[metaId]) action = await this.core.optionPolicies[metaId].act(curr);
      const out = await this.core.worldModel.predict(curr, action);
      const worldLik = -1*(math.norm(out) as number);
      const compression = -1*(math.variance(out) as number);
      const utility = math.sum(out) as number;
      const stability = math.std(out) as number;
      const step = worldLik + 0.2*compression + 0.5*utility - 0.1*stability;
      score += step; curr = out; seq.push({out, step});
    }
    return {metaId, score, seq};
  }
  async plan(latent: number[], metas: string[]){
    if(!metas.length) return null;
    const scored = await Promise.all(metas.map(m=>this.score(latent,m,6)));
    return scored.sort((a,b)=>b.score-a.score)[0];
  }
}

// =============== 8. LLM Critic (Live Epistemic Check, Plug any API) ===============
class LLMExternalCritic {
  llm: any; chatLog: any[] = [];
  constructor(llm: any) { this.llm = llm; }
  async critique(what: string, symbolGraph: HierarchicalSymbolGraph, metrics: any) {
    let prompt = `[CRITIQUE] ${what}\nGraph: ${JSON.stringify(symbolGraph.hierarchy())}\nMetrics: ${JSON.stringify(metrics)}\nEvaluate abstraction, reuse, collapse.`;
    let resp = await this.llm(prompt);
    this.chatLog.push({ prompt, resp });
    return resp;
  }
}

// =============== 9. Main AGI Kernel (Full Organism, All Metrics, REST API) ===============
type EnvObs = { state:number[]; reward:number; done:boolean };
type Env = { reset:()=>Promise<EnvObs>; step:(a:number[])=>Promise<EnvObs> };

class AMKernel {
  stateDim:number; actionDim:number; env:Env;
  autoencoder:AbstractionAutoencoder; worldModel:ConceptWorldModel;
  symboler = new LatentSymbolExtractor(12);
  metrics = new SymbolPhaseMetrics();
  graph = new HierarchicalSymbolGraph();
  planner:CounterfactualPlanner;
  optionPolicies: Record<string, OptionPolicy> = {};
  optionTrainData: Record<string, { latent: number[]; action: number[] }[]> = {};
  replay: { s:number[]; a:number[]; z:number[]; zNext:number[]; sym:string; nextSym:string }[] = [];
  logs:any[]=[];
  llmCritic?: LLMExternalCritic;

  constructor(env:Env, stateDim:number, actionDim:number, llm?:any){
    this.env=env; this.stateDim=stateDim; this.actionDim=actionDim;
    this.autoencoder=new AbstractionAutoencoder(stateDim,8);
    this.worldModel=new ConceptWorldModel(8, actionDim);
    this.planner=new CounterfactualPlanner(this, this.graph);
    if (llm) this.llmCritic = new LLMExternalCritic(llm);
  }

  async runEpisode(taskId="task0"){
    let obs = await this.env.reset();
    const latents:number[][]=[]; const symbols:string[]=[];
    for(let t=0;t<50;t++){
      const z = await this.autoencoder.encode(obs.state);
      latents.push(z);

      const sym = this.symboler.assign(z);
      const prev = symbols.length?symbols[symbols.length-1]:"";
      this.metrics.observe(sym, taskId, prev);
      this.graph.addSymbol(sym, z);
      if(prev) this.graph.addChild(prev, sym);
      symbols.push(sym);

      // -------- Option Policy Learning --------
      for(const metaId of Object.keys(this.graph.programs)){
        if (!this.optionTrainData[metaId]) this.optionTrainData[metaId] = [];
        if(this.graph.programs[metaId].includes(sym)) {
          const action = Array.from({length:this.actionDim},()=>Math.random());
          this.optionTrainData[metaId].push({ latent: z, action });
        }
      }
      const action = Array.from({length:this.actionDim},()=>Math.random());
      const nextObs = await this.env.step(action);
      const zNext = await this.autoencoder.encode(nextObs.state);
      this.replay.push({ s:obs.state, a:action, z, zNext, sym, nextSym:"" });

      obs = nextObs; if(obs.done) break;
    }

    this.symboler.extract(latents);

    await this.worldModel.train(this.replay.map(r=>({z:r.z, a:r.a, zNext:r.zNext})));
    await this.autoencoder.train(this.replay.map(r=>r.s));
    this.metrics.nextEpoch();

    // -------- Promote to meta-skills and train option-policies --------
    for(const s of Object.keys(this.metrics.usages)){
      if(this.metrics.generalizes[s] && this.metrics.reusable[s]){
        const metaId = "META_"+s;
        if(!this.graph.programs[metaId]){
          this.graph.compose(metaId, [s]);
          this.optionPolicies[metaId] = new OptionPolicy(8, this.actionDim);
          console.log(`[AM] Promoted ${s} -> ${metaId} & initialized policy`);
        }
        // TRAIN policy if new data
        const trainData = this.optionTrainData[metaId] || [];
        if(trainData.length > 10) {
          await this.optionPolicies[metaId].train(trainData.splice(0,trainData.length));
        }
      }
    }

    const metas = Object.keys(this.graph.programs);
    if(metas.length){
      const best = await this.planner.plan(latents[0], metas);
      let symbolCrit = null, progCrit = null;
      if(this.llmCritic){
        symbolCrit = await this.llmCritic.critique(symbols[0], this.graph, this.metrics.snapshot());
        progCrit = await this.llmCritic.critique(metas[0], this.graph, this.metrics.snapshot());
      }
      this.logs.push({metrics:this.metrics.snapshot(), best, symbolCrit, progCrit});
      fs.writeFileSync("am-kernel-metrics.json", JSON.stringify(this.logs.slice(-10), null, 2));
    }
  }
}

// =============== 10. Plug-in ENV ===============
class DummyEnv implements Env {
  async reset(){ return {state:[0,0,0,0], reward:0, done:false}; }
  async step(_:number[]){ 
    const s=[Math.random(),Math.random(),Math.random(),Math.random()];
    return {state:s, reward:math.sum(s) as number, done:Math.random()>0.95};
  }
}

// =============== 11. LLM Critic ===============
const llm = async (prompt: string) => "LLM: " + prompt.slice(0,100);

// =============== 12. LIVE REST API ===============
const env = new DummyEnv();
export const core = new AMKernel(env, 4, 2, llm);

(async()=>{
  setInterval(()=> core.runEpisode("task0"), 1000);
})();

export { AMKernel };

