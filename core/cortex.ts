// =======================
// AM_CORTEX (TypeScript/NodeJS)
// =======================
import * as math from "mathjs"

class SymbolPhaseMetrics {
    usages: Record<string, number> = {}
    inTasks: Record<string, Set<string>> = {}
    transitions: Record<string, Set<string>> = {}
    birthEpoch: Record<string, number> = {}
    collapseEpoch: Record<string, number> = {}
    generalization: Record<string, boolean> = {}
    reusable: Record<string, boolean> = {}
    collapsed: Record<string, boolean> = {}
    epoch = 0

    observe(symbolId: string, taskId: string, nextSymbol: string) {
        this.usages[symbolId] = (this.usages[symbolId] || 0) + 1
        if (!this.inTasks[symbolId]) this.inTasks[symbolId] = new Set()
        this.inTasks[symbolId].add(taskId)
        if (!this.transitions[symbolId]) this.transitions[symbolId] = new Set()
        this.transitions[symbolId].add(nextSymbol)
        if (!(symbolId in this.birthEpoch)) this.birthEpoch[symbolId] = this.epoch
        if (this.inTasks[symbolId].size > 1 && this.usages[symbolId] > 10 && this.transitions[symbolId].size > 3)
            this.generalization[symbolId] = true
        if (this.inTasks[symbolId].size > 2 && this.usages[symbolId] > 7)
            this.reusable[symbolId] = true
    }
    markCollapse(symbolId: string) {
        this.collapsed[symbolId] = true
        this.collapseEpoch[symbolId] = this.epoch
    }
    snapshot() {
        let out = []
        for (let id in this.usages) out.push({
            symbol: id,
            usage: this.usages[id],
            generalizes: !!this.generalization[id],
            reusable: !!this.reusable[id],
            collapsed: !!this.collapsed[id],
            tasks: [...(this.inTasks[id] || [])],
            birthEpoch: this.birthEpoch[id] || null,
            collapseEpoch: this.collapseEpoch[id] || null
        })
        return out
    }
    nextEpoch() { this.epoch++ }
}

class SymbolNode {
    id: string; vector: number[]; children: Set<string>; parent: string|null
    constructor(id: string, vector: number[]) {
        this.id = id; this.vector = vector; this.children = new Set(); this.parent = null
    }
}
class HierarchicalSymbolGraph {
    nodes: Record<string, SymbolNode> = {}
    programs: Record<string, string[]> = {}

    addSymbol(id: string, vector: number[]) {
        if (!this.nodes[id]) this.nodes[id] = new SymbolNode(id, vector)
    }
    addChild(parentId: string, childId: string) {
        if (this.nodes[parentId] && this.nodes[childId]) {
            this.nodes[parentId].children.add(childId)
            this.nodes[childId].parent = parentId
        }
    }
    composeProgram(metaId: string, sequence: string[]) { this.programs[metaId] = sequence }
    getHierarchy() {
        let out: any = {}
        for (let id in this.nodes) out[id] = { parent: this.nodes[id].parent, children: [...this.nodes[id].children] }
        return out
    }
    execute(metaId: string, input: any) {
        if (!(metaId in this.programs)) return input
        let out = input
        for (let cid of this.programs[metaId]) out = this.execute(cid, out)
        return out
    }
}

class CounterfactualPlanner {
    core: any; // should be wired to kernel/worldmodel
    symbolGraph: HierarchicalSymbolGraph
    constructor(core: any, symbolGraph: HierarchicalSymbolGraph) {
        this.core = core; this.symbolGraph = symbolGraph
    }
    scoreFuture(latent: number[], programId: string, steps=5) {
        let seq = [], score = 0, curr = latent
        for (let i=0; i<steps; ++i) {
            let out = curr // TODO: plug in real world model for out
            let worldLikelihood = -1 * (math.norm(out) as number)
            let compression = -1 * (math.variance(out) as number)
            let utility = math.sum(out)
            let stability = math.std(out) as number
            let stepScore = worldLikelihood + 0.2*compression + 0.5*utility - 0.1*stability
            score += stepScore
            curr = out; seq.push({out, stepScore})
        }
        return { program: programId, sequence: seq, score }
    }
    plan(latent: number[], programs: string[], steps=5) {
        let results = programs.map(pid => this.scoreFuture(latent, pid, steps))
        let best = results.sort((a, b) => b.score - a.score)[0]
        return best
    }
}

class LanguageCritiqueLoop {
    llm: any; chatLog: any[] = []
    constructor(llm: any) { this.llm = llm }
    async critiqueSymbol(symbolId: string, symbolGraph: HierarchicalSymbolGraph, metrics: any) {
        let prompt = `Critique concept "${symbolId}"\nGraph:\n${JSON.stringify(symbolGraph.getHierarchy())}\nMetrics:\n${JSON.stringify(metrics)}`
        let resp = await this.llm.call(prompt, {})
        this.chatLog.push({prompt, resp})
        return resp
    }
    async critiqueProgram(metaId: string, symbolGraph: HierarchicalSymbolGraph, metrics: any) {
        let prompt = `Critique hierarchical program "${metaId}"\nGraph:\n${JSON.stringify(symbolGraph.programs[metaId])}\nMetrics:\n${JSON.stringify(metrics)}`
        let resp = await this.llm.call(prompt, {})
        this.chatLog.push({prompt, resp})
        return resp
    }
}

export class AM_Cortex {
    private core: any
    private phase: SymbolPhaseMetrics
    private graph: HierarchicalSymbolGraph
    private planner: CounterfactualPlanner
    private langCritic: LanguageCritiqueLoop

    constructor(core: any, llm: any) {
        this.core = core
        this.phase = new SymbolPhaseMetrics()
        this.graph = new HierarchicalSymbolGraph()
        this.planner = new CounterfactualPlanner(core, this.graph)
        this.langCritic = new LanguageCritiqueLoop(llm)
    }
    observeSymbol(symbol: string, task: string, nextSymbol: string, vector: number[]) {
        this.phase.observe(symbol, task, nextSymbol)
        this.graph.addSymbol(symbol, vector)
    }
    composeMeta(symbols: string[]) {
        const metaId = "M" + Math.random().toString(36).slice(2,8)
        this.graph.composeProgram(metaId, symbols)
        return metaId
    }
    plan(latent: number[]) {
        let programs = Object.keys(this.graph.programs)
        if (programs.length === 0) return null
        return this.planner.plan(latent, programs, 5)
    }
    async critiqueSymbol(symbol: string) {
        return await this.langCritic.critiqueSymbol(symbol, this.graph, this.phase.snapshot())
    }
    async critiqueProgram(metaId: string) {
        return await this.langCritic.critiqueProgram(metaId, this.graph, this.phase.snapshot())
    }
    getHierarchy() { return this.graph.getHierarchy() }
    getPhaseMetrics() { return this.phase.snapshot() }
}
