/**
 * AM Core - Multi-Domain Learning Agent
 *
 * A comprehensive AGI-like cognitive architecture featuring:
 * - Hierarchical memory (episodic, semantic, skill, value)
 * - Dreamer-style world model with imagination
 * - Lifelong learning with transfer capabilities
 * - Multi-agent inner debate for reasoning
 * - Meta-cognitive self-reflection
 * - Autonomous skill discovery
 * - Curiosity-driven exploration
 * - Tool-use reasoning
 * - Long-horizon goal tracking
 * - Consciousness-like self-monitoring
 */

// Legacy exports for backwards compatibility
export { AM_Organism } from "./organism";
export { AM_Cortex } from "./cortex";

// Memory System
export {
  HierarchicalMemorySystem,
  EpisodicMemory,
  SemanticMemory,
  SkillMemory,
  ValueMemory,
} from "./memory";

// World Model
export {
  DreamerWorldModel,
  RecurrentStateSpaceModel,
  RSSM,
  ImaginationEngine,
  CounterfactualReasoner,
} from "./world";

// Lifelong Learning
export {
  LifelongLearner,
  ElasticWeightConsolidation,
  ProgressiveNeuralNetwork,
  DomainTransferManager,
  CurriculumManager,
  PrioritizedExperienceReplay,
} from "./learning";

// Multi-Agent Debate
export {
  InnerDebateOrchestrator,
  PlannerAgent,
  CriticAgent,
  VerifierAgent,
  ExplainerAgent,
  ExplorerAgent,
} from "./agents";

// Meta-Cognitive Reflection
export {
  MetaCognitiveLoop,
  DecisionTracker,
  SelfCritic,
  FailurePatternDetector,
  StrategyAdjuster,
  ArchitectureProposer,
  CompetenceTracker,
} from "./reflection";

// Skill Discovery
export {
  SkillDiscoveryEngine,
  BehaviorBuffer,
  BehaviorEncoder,
  SkillClusterer,
  SkillSynthesizer,
  SkillLibrary,
} from "./skills";

// Curiosity System
export {
  CuriosityEngine,
  RandomNetworkDistillation,
  InformationGainEstimator,
  ConceptFormationEngine,
  AbstractionReward,
} from "./curiosity";

// Tool Reasoning
export {
  ToolReasoningEngine,
  ToolRegistry,
  PermissionManager,
  ToolSelector,
  ToolChainBuilder,
  ToolExecutorEngine,
} from "./tools";

// Goal Tracking
export {
  GoalTracker,
  GoalRepository,
  GoalDecomposer,
  ProgressEstimator,
  StrategyReviser,
} from "./goals";

// Consciousness Monitoring
export {
  ConsciousnessMonitor,
  ConfidenceMonitor,
  UncertaintyMonitor,
  CompetenceMonitor,
  AlignmentMonitor,
  MotivationMonitor,
  AttentionMonitor,
  MetacognitionMonitor,
} from "./consciousness";

// Multi-Domain Agent (Main Export)
export {
  MultiDomainAgent,
  createMultiDomainAgent,
  AgentConfig,
  AgentState,
  StepResult,
  EpisodeResult,
  ReflectionResult,
} from "./multi_domain_agent";

// ==================== SERVER ENTRY POINT ====================

import express, { Request, Response } from "express";
import * as math from "mathjs";
import { AM_Organism } from "./organism";
import { AM_Cortex } from "./cortex";
import { createMultiDomainAgent } from "./multi_domain_agent";

// Legacy environment for backwards compatibility
class DummyEnv {
  async reset() {
    return { state: [0, 0, 0, 0], reward: 0, done: false };
  }
  async step(_: number[]) {
    const s = [Math.random(), Math.random(), Math.random(), Math.random()];
    return { state: s, reward: math.sum(s) as number, done: Math.random() > 0.95 };
  }
}

// Boot legacy organism + cortex
const env = new DummyEnv();
const organism = new AM_Organism(4, 2);
const cortex = new AM_Cortex(organism);

// Boot new multi-domain agent
const agent = createMultiDomainAgent("AM-Agent", {
  latentDim: 32,
  beliefDim: 256,
  actionDim: 4,
  inputDim: 64,
});

// Autonomous symbol genesis (legacy)
setInterval(async () => {
  const obs = await env.reset();
  const z = organism.encode(obs.state);
  const sym = "S_" + Math.floor(math.sum(z) * 100);
  cortex.observe(sym, "task0", "", z);
}, 1000);

// REST API
const app = express();
app.use(express.json());

// Legacy endpoints
app.get("/metrics", (req: Request, res: Response) => {
  res.json(cortex.metrics());
});

app.get("/hierarchy", (req: Request, res: Response) => {
  res.json(cortex.hierarchy());
});

// New multi-domain agent endpoints
app.get("/agent/state", (req: Request, res: Response) => {
  res.json(agent.getState());
});

app.get("/agent/stats", (req: Request, res: Response) => {
  res.json(agent.getStats());
});

app.get("/agent/consciousness", (req: Request, res: Response) => {
  res.json(agent.getConsciousnessState());
});

app.get("/agent/introspect", (req: Request, res: Response) => {
  res.json(agent.introspect());
});

app.post("/agent/domain", async (req: Request, res: Response) => {
  const { name, sampleData } = req.body;
  const domainId = await agent.enterDomain(name, sampleData);
  res.json({ domainId });
});

app.post("/agent/goal", (req: Request, res: Response) => {
  const { name, description, estimatedEffort, priority, deadline } = req.body;
  const goalId = agent.createGoal(name, description, estimatedEffort, { priority, deadline });
  res.json({ goalId });
});

app.post("/agent/goal/:id/activate", (req: Request, res: Response) => {
  const result = agent.activateGoal(req.params.id);
  res.json({ success: result });
});

app.post("/agent/reflect", async (req: Request, res: Response) => {
  const result = await agent.reflect();
  res.json(result);
});

app.get("/agent/tools", (req: Request, res: Response) => {
  res.json(agent.getAvailableTools());
});

app.get("/agent/skills", (req: Request, res: Response) => {
  const state = req.query.state
    ? JSON.parse(req.query.state as string)
    : Array(64).fill(0);
  res.json(agent.getApplicableSkills(state));
});

const PORT = process.env.PORT || 5030;

app.listen(PORT, () => {
  console.log(`🧠 AM Multi-Domain Agent online at http://localhost:${PORT}`);
  console.log(`   Agent: ${agent.name}`);
  console.log(`   Endpoints:`);
  console.log(`     GET  /metrics          - Legacy cortex metrics`);
  console.log(`     GET  /hierarchy        - Legacy symbol hierarchy`);
  console.log(`     GET  /agent/state      - Agent state`);
  console.log(`     GET  /agent/stats      - Agent statistics`);
  console.log(`     GET  /agent/consciousness - Consciousness state`);
  console.log(`     GET  /agent/introspect - Self-introspection`);
  console.log(`     POST /agent/domain     - Enter new domain`);
  console.log(`     POST /agent/goal       - Create goal`);
  console.log(`     POST /agent/reflect    - Trigger reflection`);
  console.log(`     GET  /agent/tools      - Available tools`);
  console.log(`     GET  /agent/skills     - Applicable skills`);
});
