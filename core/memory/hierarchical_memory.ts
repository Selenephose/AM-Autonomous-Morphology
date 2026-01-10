/**
 * Hierarchical Memory System for Multi-Domain Learning Agent
 *
 * Four memory types:
 * - Episodic: Event sequences with temporal context
 * - Semantic: Abstracted knowledge and concepts
 * - Skill: Procedural programs and action sequences
 * - Value: Outcome-based learning (what worked/didn't)
 */

import * as math from "mathjs";

// ==================== TYPES ====================

export interface Episode {
  id: string;
  timestamp: number;
  domain: string;
  context: number[];
  events: EpisodeEvent[];
  outcome: EpisodeOutcome;
  importance: number;
  accessCount: number;
  lastAccessed: number;
}

export interface EpisodeEvent {
  step: number;
  state: number[];
  action: number[];
  latent: number[];
  symbol: string;
  reward: number;
  metadata?: Record<string, unknown>;
}

export interface EpisodeOutcome {
  success: boolean;
  totalReward: number;
  goalAchieved: boolean;
  lessonsLearned: string[];
}

export interface SemanticConcept {
  id: string;
  name: string;
  embedding: number[];
  relatedConcepts: string[];
  sourceEpisodes: string[];
  confidence: number;
  domain: string;
  abstractionLevel: number;
  createdAt: number;
  updatedAt: number;
}

export interface SkillProgram {
  id: string;
  name: string;
  symbolSequence: string[];
  preconditions: number[];
  postconditions: number[];
  successRate: number;
  executionCount: number;
  averageReward: number;
  domains: string[];
  transferability: number;
  createdAt: number;
}

export interface ValueMemory {
  id: string;
  context: number[];
  action: number[] | string;
  outcome: number;
  uncertainty: number;
  sampleCount: number;
  lastUpdated: number;
  domain: string;
}

// ==================== EPISODIC MEMORY ====================

export class EpisodicMemory {
  private episodes: Map<string, Episode> = new Map();
  private temporalIndex: string[] = [];
  private domainIndex: Map<string, string[]> = new Map();
  private maxEpisodes: number;
  private importanceThreshold: number;

  constructor(maxEpisodes = 10000, importanceThreshold = 0.3) {
    this.maxEpisodes = maxEpisodes;
    this.importanceThreshold = importanceThreshold;
  }

  store(episode: Omit<Episode, "id" | "accessCount" | "lastAccessed">): string {
    const id = `ep_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    const fullEpisode: Episode = {
      ...episode,
      id,
      accessCount: 0,
      lastAccessed: Date.now(),
    };

    this.episodes.set(id, fullEpisode);
    this.temporalIndex.push(id);

    const domainEps = this.domainIndex.get(episode.domain) || [];
    domainEps.push(id);
    this.domainIndex.set(episode.domain, domainEps);

    if (this.episodes.size > this.maxEpisodes) {
      this.consolidate();
    }

    return id;
  }

  retrieve(query: number[], k = 5): Episode[] {
    const scored = Array.from(this.episodes.values()).map(ep => ({
      episode: ep,
      similarity: this.cosineSimilarity(query, ep.context),
    }));

    scored.sort((a, b) => b.similarity - a.similarity);

    const results = scored.slice(0, k).map(s => {
      s.episode.accessCount++;
      s.episode.lastAccessed = Date.now();
      return s.episode;
    });

    return results;
  }

  retrieveByDomain(domain: string, limit = 10): Episode[] {
    const ids = this.domainIndex.get(domain) || [];
    return ids
      .slice(-limit)
      .map(id => this.episodes.get(id)!)
      .filter(Boolean);
  }

  retrieveRecent(n = 10): Episode[] {
    return this.temporalIndex
      .slice(-n)
      .map(id => this.episodes.get(id)!)
      .filter(Boolean)
      .reverse();
  }

  consolidate(): void {
    const episodes = Array.from(this.episodes.values());

    episodes.sort((a, b) => {
      const scoreA = this.consolidationScore(a);
      const scoreB = this.consolidationScore(b);
      return scoreA - scoreB;
    });

    const toRemove = Math.floor(this.maxEpisodes * 0.2);
    for (let i = 0; i < toRemove && i < episodes.length; i++) {
      const ep = episodes[i];
      if (ep.importance < this.importanceThreshold) {
        this.episodes.delete(ep.id);
        this.temporalIndex = this.temporalIndex.filter(id => id !== ep.id);
        const domainEps = this.domainIndex.get(ep.domain) || [];
        this.domainIndex.set(ep.domain, domainEps.filter(id => id !== ep.id));
      }
    }
  }

  private consolidationScore(ep: Episode): number {
    const recency = (Date.now() - ep.lastAccessed) / (1000 * 60 * 60 * 24);
    const frequency = Math.log(ep.accessCount + 1);
    return ep.importance * 0.4 + frequency * 0.3 - recency * 0.01 + (ep.outcome.success ? 0.2 : 0);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  size(): number {
    return this.episodes.size;
  }

  getDomains(): string[] {
    return Array.from(this.domainIndex.keys());
  }

  getStats(): Record<string, unknown> {
    const episodes = Array.from(this.episodes.values());
    return {
      totalEpisodes: episodes.length,
      domains: this.getDomains(),
      averageImportance: episodes.reduce((s, e) => s + e.importance, 0) / (episodes.length || 1),
      successRate: episodes.filter(e => e.outcome.success).length / (episodes.length || 1),
    };
  }
}

// ==================== SEMANTIC MEMORY ====================

export class SemanticMemory {
  private concepts: Map<string, SemanticConcept> = new Map();
  private embeddings: Map<string, number[]> = new Map();
  private domainIndex: Map<string, string[]> = new Map();

  store(concept: Omit<SemanticConcept, "id" | "createdAt" | "updatedAt">): string {
    const id = `sem_${concept.name.replace(/\s+/g, "_")}_${Math.random().toString(36).slice(2, 6)}`;

    const fullConcept: SemanticConcept = {
      ...concept,
      id,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.concepts.set(id, fullConcept);
    this.embeddings.set(id, concept.embedding);

    const domainConcepts = this.domainIndex.get(concept.domain) || [];
    domainConcepts.push(id);
    this.domainIndex.set(concept.domain, domainConcepts);

    return id;
  }

  retrieve(query: number[], k = 5): SemanticConcept[] {
    const scored = Array.from(this.concepts.values()).map(concept => ({
      concept,
      similarity: this.cosineSimilarity(query, concept.embedding),
    }));

    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, k).map(s => s.concept);
  }

  retrieveByName(name: string): SemanticConcept | undefined {
    return Array.from(this.concepts.values()).find(c =>
      c.name.toLowerCase().includes(name.toLowerCase())
    );
  }

  update(id: string, updates: Partial<SemanticConcept>): boolean {
    const concept = this.concepts.get(id);
    if (!concept) return false;

    Object.assign(concept, updates, { updatedAt: Date.now() });
    if (updates.embedding) {
      this.embeddings.set(id, updates.embedding);
    }
    return true;
  }

  linkConcepts(id1: string, id2: string): void {
    const c1 = this.concepts.get(id1);
    const c2 = this.concepts.get(id2);
    if (c1 && c2) {
      if (!c1.relatedConcepts.includes(id2)) c1.relatedConcepts.push(id2);
      if (!c2.relatedConcepts.includes(id1)) c2.relatedConcepts.push(id1);
    }
  }

  abstract(episodes: Episode[], domain: string): SemanticConcept | null {
    if (episodes.length < 2) return null;

    const avgEmbedding = this.averageEmbeddings(episodes.map(e => e.context));
    const lessons = episodes.flatMap(e => e.outcome.lessonsLearned);
    const uniqueLessons = [...new Set(lessons)];

    const name = `concept_${domain}_${Date.now()}`;

    const id = this.store({
      name,
      embedding: avgEmbedding,
      relatedConcepts: [],
      sourceEpisodes: episodes.map(e => e.id),
      confidence: Math.min(0.3 + episodes.length * 0.1, 1.0),
      domain,
      abstractionLevel: 1,
    });

    return this.concepts.get(id)!;
  }

  private averageEmbeddings(embeddings: number[][]): number[] {
    if (embeddings.length === 0) return [];
    const dim = embeddings[0].length;
    const avg = new Array(dim).fill(0);
    for (const emb of embeddings) {
      for (let i = 0; i < dim; i++) {
        avg[i] += emb[i] / embeddings.length;
      }
    }
    return avg;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  getByDomain(domain: string): SemanticConcept[] {
    const ids = this.domainIndex.get(domain) || [];
    return ids.map(id => this.concepts.get(id)!).filter(Boolean);
  }

  size(): number {
    return this.concepts.size;
  }

  getStats(): Record<string, unknown> {
    const concepts = Array.from(this.concepts.values());
    return {
      totalConcepts: concepts.length,
      domains: Array.from(this.domainIndex.keys()),
      avgConfidence: concepts.reduce((s, c) => s + c.confidence, 0) / (concepts.length || 1),
      avgAbstractionLevel: concepts.reduce((s, c) => s + c.abstractionLevel, 0) / (concepts.length || 1),
    };
  }
}

// ==================== SKILL MEMORY ====================

export class SkillMemory {
  private skills: Map<string, SkillProgram> = new Map();
  private domainIndex: Map<string, string[]> = new Map();
  private symbolIndex: Map<string, string[]> = new Map();

  store(skill: Omit<SkillProgram, "id" | "executionCount" | "createdAt">): string {
    const id = `skill_${skill.name.replace(/\s+/g, "_")}_${Math.random().toString(36).slice(2, 6)}`;

    const fullSkill: SkillProgram = {
      ...skill,
      id,
      executionCount: 0,
      createdAt: Date.now(),
    };

    this.skills.set(id, fullSkill);

    for (const domain of skill.domains) {
      const domainSkills = this.domainIndex.get(domain) || [];
      domainSkills.push(id);
      this.domainIndex.set(domain, domainSkills);
    }

    for (const symbol of skill.symbolSequence) {
      const symbolSkills = this.symbolIndex.get(symbol) || [];
      symbolSkills.push(id);
      this.symbolIndex.set(symbol, symbolSkills);
    }

    return id;
  }

  retrieve(context: number[], k = 5): SkillProgram[] {
    const scored = Array.from(this.skills.values()).map(skill => ({
      skill,
      score: this.matchScore(context, skill),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k).map(s => s.skill);
  }

  retrieveBySymbol(symbol: string): SkillProgram[] {
    const ids = this.symbolIndex.get(symbol) || [];
    return ids.map(id => this.skills.get(id)!).filter(Boolean);
  }

  retrieveByDomain(domain: string): SkillProgram[] {
    const ids = this.domainIndex.get(domain) || [];
    return ids.map(id => this.skills.get(id)!).filter(Boolean);
  }

  retrieveTransferable(minTransferability = 0.5): SkillProgram[] {
    return Array.from(this.skills.values())
      .filter(s => s.transferability >= minTransferability)
      .sort((a, b) => b.transferability - a.transferability);
  }

  recordExecution(id: string, success: boolean, reward: number): void {
    const skill = this.skills.get(id);
    if (!skill) return;

    skill.executionCount++;
    skill.successRate = (skill.successRate * (skill.executionCount - 1) + (success ? 1 : 0)) / skill.executionCount;
    skill.averageReward = (skill.averageReward * (skill.executionCount - 1) + reward) / skill.executionCount;

    this.updateTransferability(id);
  }

  private updateTransferability(id: string): void {
    const skill = this.skills.get(id);
    if (!skill) return;

    const domainDiversity = skill.domains.length / Math.max(this.domainIndex.size, 1);
    const reliability = skill.successRate * Math.min(skill.executionCount / 10, 1);
    skill.transferability = (domainDiversity * 0.4 + reliability * 0.6);
  }

  private matchScore(context: number[], skill: SkillProgram): number {
    const preconditionMatch = this.cosineSimilarity(context, skill.preconditions);
    const reliabilityBonus = skill.successRate * Math.log(skill.executionCount + 1) * 0.1;
    return preconditionMatch + reliabilityBonus;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  merge(skill1Id: string, skill2Id: string): SkillProgram | null {
    const s1 = this.skills.get(skill1Id);
    const s2 = this.skills.get(skill2Id);
    if (!s1 || !s2) return null;

    const mergedSequence = [...new Set([...s1.symbolSequence, ...s2.symbolSequence])];
    const mergedDomains = [...new Set([...s1.domains, ...s2.domains])];

    const avgPreconditions = s1.preconditions.map((v, i) =>
      (v + (s2.preconditions[i] || 0)) / 2
    );
    const avgPostconditions = s1.postconditions.map((v, i) =>
      (v + (s2.postconditions[i] || 0)) / 2
    );

    const id = this.store({
      name: `merged_${s1.name}_${s2.name}`,
      symbolSequence: mergedSequence,
      preconditions: avgPreconditions,
      postconditions: avgPostconditions,
      successRate: (s1.successRate + s2.successRate) / 2,
      averageReward: (s1.averageReward + s2.averageReward) / 2,
      domains: mergedDomains,
      transferability: Math.max(s1.transferability, s2.transferability),
    });

    return this.skills.get(id)!;
  }

  size(): number {
    return this.skills.size;
  }

  getStats(): Record<string, unknown> {
    const skills = Array.from(this.skills.values());
    return {
      totalSkills: skills.length,
      domains: Array.from(this.domainIndex.keys()),
      avgSuccessRate: skills.reduce((s, sk) => s + sk.successRate, 0) / (skills.length || 1),
      avgTransferability: skills.reduce((s, sk) => s + sk.transferability, 0) / (skills.length || 1),
      totalExecutions: skills.reduce((s, sk) => s + sk.executionCount, 0),
    };
  }
}

// ==================== VALUE MEMORY ====================

export class ValueMemory {
  private values: Map<string, ValueMemory> = new Map();
  private contextIndex: number[][] = [];
  private valueIds: string[] = [];
  private latentDim: number;

  constructor(latentDim = 8) {
    this.latentDim = latentDim;
  }

  store(context: number[], action: number[] | string, outcome: number, domain: string): string {
    const existingId = this.findSimilar(context, action);

    if (existingId) {
      const existing = this.values.get(existingId) as unknown as ValueMemoryEntry;
      existing.sampleCount++;
      const alpha = 1 / existing.sampleCount;
      existing.outcome = existing.outcome * (1 - alpha) + outcome * alpha;
      existing.uncertainty = Math.max(0.1, existing.uncertainty * 0.95);
      existing.lastUpdated = Date.now();
      return existingId;
    }

    const id = `val_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    const entry: ValueMemoryEntry = {
      id,
      context,
      action,
      outcome,
      uncertainty: 1.0,
      sampleCount: 1,
      lastUpdated: Date.now(),
      domain,
    };

    this.values.set(id, entry as unknown as ValueMemory);
    this.contextIndex.push(context);
    this.valueIds.push(id);

    return id;
  }

  query(context: number[], action: number[] | string): { value: number; uncertainty: number } | null {
    const similar = this.findSimilar(context, action);
    if (!similar) return null;

    const entry = this.values.get(similar) as unknown as ValueMemoryEntry;
    return {
      value: entry.outcome,
      uncertainty: entry.uncertainty,
    };
  }

  queryBest(context: number[], actions: (number[] | string)[]): { action: number[] | string; value: number } | null {
    let best: { action: number[] | string; value: number } | null = null;

    for (const action of actions) {
      const result = this.query(context, action);
      if (result && (!best || result.value > best.value)) {
        best = { action, value: result.value };
      }
    }

    return best;
  }

  private findSimilar(context: number[], action: number[] | string): string | null {
    const threshold = 0.9;

    for (let i = 0; i < this.contextIndex.length; i++) {
      const sim = this.cosineSimilarity(context, this.contextIndex[i]);
      if (sim >= threshold) {
        const entry = this.values.get(this.valueIds[i]) as unknown as ValueMemoryEntry;
        if (this.actionsMatch(action, entry.action)) {
          return this.valueIds[i];
        }
      }
    }

    return null;
  }

  private actionsMatch(a1: number[] | string, a2: number[] | string): boolean {
    if (typeof a1 === "string" && typeof a2 === "string") {
      return a1 === a2;
    }
    if (Array.isArray(a1) && Array.isArray(a2)) {
      return this.cosineSimilarity(a1, a2) > 0.95;
    }
    return false;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dotProduct / (normA * normB) : 0;
  }

  getHighValueActions(context: number[], threshold = 0.5): ValueMemoryEntry[] {
    const results: ValueMemoryEntry[] = [];

    for (let i = 0; i < this.contextIndex.length; i++) {
      const sim = this.cosineSimilarity(context, this.contextIndex[i]);
      if (sim >= 0.7) {
        const entry = this.values.get(this.valueIds[i]) as unknown as ValueMemoryEntry;
        if (entry.outcome >= threshold) {
          results.push(entry);
        }
      }
    }

    return results.sort((a, b) => b.outcome - a.outcome);
  }

  prune(maxSize = 10000): number {
    if (this.values.size <= maxSize) return 0;

    const entries = Array.from(this.values.entries()) as unknown as [string, ValueMemoryEntry][];
    entries.sort((a, b) => {
      const scoreA = a[1].outcome * Math.log(a[1].sampleCount + 1) / (a[1].uncertainty + 0.1);
      const scoreB = b[1].outcome * Math.log(b[1].sampleCount + 1) / (b[1].uncertainty + 0.1);
      return scoreB - scoreA;
    });

    const toKeep = entries.slice(0, maxSize);
    const removedCount = entries.length - maxSize;

    this.values.clear();
    this.contextIndex = [];
    this.valueIds = [];

    for (const [id, entry] of toKeep) {
      this.values.set(id, entry as unknown as ValueMemory);
      this.contextIndex.push(entry.context);
      this.valueIds.push(id);
    }

    return removedCount;
  }

  size(): number {
    return this.values.size;
  }

  getStats(): Record<string, unknown> {
    const entries = Array.from(this.values.values()) as unknown as ValueMemoryEntry[];
    return {
      totalEntries: entries.length,
      avgOutcome: entries.reduce((s, e) => s + e.outcome, 0) / (entries.length || 1),
      avgUncertainty: entries.reduce((s, e) => s + e.uncertainty, 0) / (entries.length || 1),
      avgSampleCount: entries.reduce((s, e) => s + e.sampleCount, 0) / (entries.length || 1),
    };
  }
}

interface ValueMemoryEntry {
  id: string;
  context: number[];
  action: number[] | string;
  outcome: number;
  uncertainty: number;
  sampleCount: number;
  lastUpdated: number;
  domain: string;
}

// ==================== UNIFIED HIERARCHICAL MEMORY ====================

export class HierarchicalMemorySystem {
  episodic: EpisodicMemory;
  semantic: SemanticMemory;
  skills: SkillMemory;
  values: ValueMemory;

  private consolidationInterval: number;
  private lastConsolidation: number = 0;

  constructor(config: {
    maxEpisodes?: number;
    latentDim?: number;
    consolidationInterval?: number;
  } = {}) {
    this.episodic = new EpisodicMemory(config.maxEpisodes || 10000);
    this.semantic = new SemanticMemory();
    this.skills = new SkillMemory();
    this.values = new ValueMemory(config.latentDim || 8);
    this.consolidationInterval = config.consolidationInterval || 1000 * 60 * 5;
  }

  async consolidate(): Promise<ConsolidationReport> {
    const now = Date.now();
    if (now - this.lastConsolidation < this.consolidationInterval) {
      return { skipped: true, reason: "Too soon since last consolidation" };
    }

    this.lastConsolidation = now;
    const report: ConsolidationReport = { skipped: false };

    report.episodicPruned = this.consolidateEpisodic();
    report.conceptsCreated = this.abstractToSemantic();
    report.skillsMerged = this.consolidateSkills();
    report.valuesPruned = this.values.prune();

    return report;
  }

  private consolidateEpisodic(): number {
    const beforeSize = this.episodic.size();
    this.episodic.consolidate();
    return beforeSize - this.episodic.size();
  }

  private abstractToSemantic(): number {
    let conceptsCreated = 0;

    for (const domain of this.episodic.getDomains()) {
      const episodes = this.episodic.retrieveByDomain(domain, 20);

      const successfulEpisodes = episodes.filter(e => e.outcome.success);
      if (successfulEpisodes.length >= 3) {
        const concept = this.semantic.abstract(successfulEpisodes, domain);
        if (concept) conceptsCreated++;
      }
    }

    return conceptsCreated;
  }

  private consolidateSkills(): number {
    let mergeCount = 0;
    const skills = this.skills.retrieveTransferable(0.3);

    for (let i = 0; i < skills.length - 1; i++) {
      for (let j = i + 1; j < skills.length; j++) {
        const overlap = this.skillOverlap(skills[i], skills[j]);
        if (overlap > 0.7 && overlap < 0.95) {
          const merged = this.skills.merge(skills[i].id, skills[j].id);
          if (merged) mergeCount++;
        }
      }
    }

    return mergeCount;
  }

  private skillOverlap(s1: SkillProgram, s2: SkillProgram): number {
    const set1 = new Set(s1.symbolSequence);
    const set2 = new Set(s2.symbolSequence);
    const intersection = [...set1].filter(x => set2.has(x)).length;
    const union = new Set([...s1.symbolSequence, ...s2.symbolSequence]).size;
    return intersection / union;
  }

  getFullStats(): Record<string, unknown> {
    return {
      episodic: this.episodic.getStats(),
      semantic: this.semantic.getStats(),
      skills: this.skills.getStats(),
      values: this.values.getStats(),
      lastConsolidation: this.lastConsolidation,
    };
  }

  exportState(): string {
    return JSON.stringify({
      stats: this.getFullStats(),
      timestamp: Date.now(),
    });
  }
}

interface ConsolidationReport {
  skipped: boolean;
  reason?: string;
  episodicPruned?: number;
  conceptsCreated?: number;
  skillsMerged?: number;
  valuesPruned?: number;
}
