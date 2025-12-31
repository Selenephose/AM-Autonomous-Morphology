// core/cortex.ts
import { AM_Organism } from "./organism"
import * as math from "mathjs"

// ================= Symbol Metrics =================
class SymbolPhaseMetrics {
  usages: Record<string, number> = {}
  tasks: Record<string, Set<string>> = {}
  transitions: Record<string, Set<string>> = {}
  reusable: Record<string, boolean> = {}
  generalizes: Record<string, boolean> = {}
  epoch = 0

  observe(sym: string, task: string, prev: string) {
    this.usages[sym] = (this.usages[sym] || 0) + 1

    this.tasks[sym] ||= new Set()
    this.tasks[sym].add(task)

    this.transitions[sym] ||= new Set()
    if (prev) this.transitions[sym].add(prev)

    if (this.usages[sym] > 10 && this.transitions[sym].size > 3)
      this.generalizes[sym] = true

    if (this.usages[sym] > 7 && this.tasks[sym].size > 2)
      this.reusable[sym] = true
  }

  snapshot() {
    return Object.keys(this.usages).map(s => ({
      symbol: s,
      usage: this.usages[s],
      reusable: !!this.reusable[s],
      generalizes: !!this.generalizes[s],
      tasks: [...(this.tasks[s] || [])],
      transitions: [...(this.transitions[s] || [])]
    }))
  }
}

// ================= Symbol Graph =================
class SymbolNode {
  id: string
  vector: number[]
  parent: string | null = null
  children = new Set<string>()
  constructor(id: string, v: number[]) { this.id = id; this.vector = v }
}

class HierarchicalSymbolGraph {
  nodes: Record<string, SymbolNode> = {}
  programs: Record<string, string[]> = {}

  addSymbol(id: string, v: number[]) {
    if (!this.nodes[id]) this.nodes[id] = new SymbolNode(id, v)
  }

  addEdge(p: string, c: string) {
    if (this.nodes[p] && this.nodes[c]) {
      this.nodes[p].children.add(c)
      this.nodes[c].parent = p
    }
  }

  compose(meta: string, seq: string[]) {
    this.programs[meta] = seq
  }

  hierarchy() {
    return Object.fromEntries(
      Object.entries(this.nodes).map(([k,n]) => [
        k,
        { parent: n.parent, children: [...n.children] }
      ])
    )
  }
}

// ================= Counterfactual Planner =================
class CounterfactualPlanner {
  organism: AM_Organism
  graph: HierarchicalSymbolGraph

  constructor(org: AM_Organism, graph: HierarchicalSymbolGraph) {
    this.organism = org
    this.graph = graph
  }

  private rollout(start: number[], meta: string, depth = 6) {
    let z = start.slice()
    let score = 0

    const seq = this.graph.programs[meta] || []

    for (let i = 0; i < depth; i++) {
      const a = Array.from({ length: this.organism.actionDim }, () => Math.random() * 2 - 1)
      z = this.organism.predict(z, a)

      const util = math.sum(z) as unknown as number
      const stab = math.std(z) as unknown as number

      score += util - 0.2 * stab
    }

    return score
  }

  plan(z: number[]) {
    const metas = Object.keys(this.graph.programs)
    if (!metas.length) return null

    return metas
      .map(m => ({ meta: m, score: this.rollout(z, m) }))
      .sort((a,b) => b.score - a.score)[0]
  }
}

// ================= Cortex =================
export class AM_Cortex {
  organism: AM_Organism
  phase = new SymbolPhaseMetrics()
  graph = new HierarchicalSymbolGraph()
  planner: CounterfactualPlanner

  constructor(org: AM_Organism) {
    this.organism = org
    this.planner = new CounterfactualPlanner(org, this.graph)
  }

  observe(sym: string, task: string, prev: string, z: number[]) {
    this.phase.observe(sym, task, prev)
    this.graph.addSymbol(sym, z)
    if (prev) this.graph.addEdge(prev, sym)
  }

  compose(symbols: string[]) {
    const id = "M_" + Math.random().toString(36).slice(2,7)
    this.graph.compose(id, symbols)
    return id
  }

  plan(z: number[]) {
    return this.planner.plan(z)
  }

  metrics() { return this.phase.snapshot() }
  hierarchy() { return this.graph.hierarchy() }
}
