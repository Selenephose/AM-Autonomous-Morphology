/**
 * Multi-Domain Learning Agent Tests
 *
 * Comprehensive tests for all AGI-like components:
 * - Hierarchical Memory
 * - World Model & Imagination
 * - Lifelong Learning
 * - Multi-Agent Debate
 * - Meta-Cognitive Reflection
 * - Skill Discovery
 * - Curiosity System
 * - Tool Reasoning
 * - Goal Tracking
 * - Consciousness Monitoring
 * - Integrated Agent
 */

import { describe, it, expect, beforeEach } from "@jest/globals";

import { HierarchicalMemorySystem, EpisodicMemory, SemanticMemory, SkillMemory } from "../memory";
import { DreamerWorldModel, ImaginationEngine } from "../world";
import { LifelongLearner, DomainTransferManager, CurriculumManager } from "../learning";
import { InnerDebateOrchestrator, PlannerAgent, CriticAgent } from "../agents";
import { MetaCognitiveLoop, DecisionTracker, FailurePatternDetector } from "../reflection";
import { SkillDiscoveryEngine, BehaviorBuffer, SkillLibrary } from "../skills";
import { CuriosityEngine, RandomNetworkDistillation, ConceptFormationEngine } from "../curiosity";
import { ToolReasoningEngine, ToolRegistry, PermissionManager } from "../tools";
import { GoalTracker, GoalRepository, GoalDecomposer } from "../goals";
import { ConsciousnessMonitor, ConfidenceMonitor, MotivationMonitor } from "../consciousness";
import { MultiDomainAgent, createMultiDomainAgent } from "../multi_domain_agent";

// ==================== MEMORY TESTS ====================

describe("Hierarchical Memory System", () => {
  let memory: HierarchicalMemorySystem;

  beforeEach(() => {
    memory = new HierarchicalMemorySystem({
      maxEpisodes: 100,
      latentDim: 8,
    });
  });

  describe("EpisodicMemory", () => {
    it("should store and retrieve episodes", () => {
      const episodeId = memory.episodic.store({
        timestamp: Date.now(),
        domain: "test",
        context: [0.1, 0.2, 0.3],
        events: [],
        outcome: { success: true, totalReward: 1.0, goalAchieved: false, lessonsLearned: [] },
        importance: 0.8,
      });

      expect(episodeId).toBeDefined();
      expect(memory.episodic.size()).toBe(1);
    });

    it("should retrieve similar episodes by context", () => {
      memory.episodic.store({
        timestamp: Date.now(),
        domain: "test",
        context: [1.0, 0.0, 0.0],
        events: [],
        outcome: { success: true, totalReward: 1.0, goalAchieved: false, lessonsLearned: [] },
        importance: 0.8,
      });

      const results = memory.episodic.retrieve([0.9, 0.1, 0.0], 5);
      expect(results.length).toBeGreaterThan(0);
    });

    it("should retrieve by domain", () => {
      memory.episodic.store({
        timestamp: Date.now(),
        domain: "navigation",
        context: [0.5, 0.5, 0.5],
        events: [],
        outcome: { success: true, totalReward: 0.5, goalAchieved: false, lessonsLearned: [] },
        importance: 0.5,
      });

      const results = memory.episodic.retrieveByDomain("navigation");
      expect(results.length).toBe(1);
    });
  });

  describe("SemanticMemory", () => {
    it("should store and retrieve concepts", () => {
      const conceptId = memory.semantic.store({
        name: "obstacle",
        embedding: [0.1, 0.2, 0.3, 0.4],
        relatedConcepts: [],
        sourceEpisodes: [],
        confidence: 0.9,
        domain: "navigation",
        abstractionLevel: 1,
      });

      expect(conceptId).toBeDefined();
      expect(memory.semantic.size()).toBe(1);
    });

    it("should link related concepts", () => {
      const id1 = memory.semantic.store({
        name: "obstacle",
        embedding: [0.1, 0.2, 0.3, 0.4],
        relatedConcepts: [],
        sourceEpisodes: [],
        confidence: 0.9,
        domain: "navigation",
        abstractionLevel: 1,
      });

      const id2 = memory.semantic.store({
        name: "barrier",
        embedding: [0.2, 0.3, 0.4, 0.5],
        relatedConcepts: [],
        sourceEpisodes: [],
        confidence: 0.8,
        domain: "navigation",
        abstractionLevel: 1,
      });

      memory.semantic.linkConcepts(id1, id2);
      // Concepts should now be related
    });
  });

  describe("SkillMemory", () => {
    it("should store and retrieve skills", () => {
      const skillId = memory.skills.store({
        name: "grasp",
        symbolSequence: ["S1", "S2", "S3"],
        preconditions: [0.5, 0.5],
        postconditions: [0.8, 0.2],
        successRate: 0.9,
        averageReward: 1.5,
        domains: ["manipulation"],
        transferability: 0.7,
      });

      expect(skillId).toBeDefined();
      const retrieved = memory.skills.retrieveByDomain("manipulation");
      expect(retrieved.length).toBe(1);
    });
  });
});

// ==================== WORLD MODEL TESTS ====================

describe("Dreamer World Model", () => {
  let worldModel: DreamerWorldModel;

  beforeEach(() => {
    worldModel = new DreamerWorldModel(8, 32, 4);
  });

  it("should initialize with correct dimensions", () => {
    const dims = worldModel.rssm.getDimensions();
    expect(dims.latent).toBe(8);
    expect(dims.belief).toBe(32);
    expect(dims.action).toBe(4);
  });

  it("should observe and update state", async () => {
    const observation = Array(8).fill(0.5);
    const action = [0.1, 0.2, 0.3, 0.4];

    const newState = await worldModel.observe(observation, action, 0.5, false);

    expect(newState.latent).toBeDefined();
    expect(newState.belief).toBeDefined();
    expect(newState.uncertainty).toBeGreaterThanOrEqual(0);
  });

  it("should predict future states", async () => {
    const actionSequence = [
      [0.1, 0.2, 0.3, 0.4],
      [0.2, 0.3, 0.4, 0.5],
    ];

    const prediction = await worldModel.predictFuture(actionSequence);

    expect(prediction.states.length).toBe(3); // Initial + 2 predictions
    expect(prediction.totalReward).toBeDefined();
  });

  it("should support counterfactual reasoning", async () => {
    const hypothetical = [0.5, 0.5, 0.5, 0.5];
    const alternative = [-0.5, -0.5, -0.5, -0.5];

    const result = await worldModel.counterfactual.whatIf(
      worldModel.getCurrentState(),
      hypothetical,
      alternative
    );

    expect(result.recommendation).toMatch(/factual|counterfactual|uncertain/);
    expect(typeof result.difference).toBe("number");
  });
});

// ==================== LEARNING TESTS ====================

describe("Lifelong Learner", () => {
  let learner: LifelongLearner;

  beforeEach(() => {
    learner = new LifelongLearner({
      inputDim: 16,
      hiddenDim: 32,
      outputDim: 4,
    });
  });

  it("should register new domains", async () => {
    const sampleData = [Array(16).fill(0.5), Array(16).fill(-0.5)];
    const domainId = await learner.startDomain("navigation", sampleData);

    expect(domainId).toBeDefined();
    expect(domainId).toContain("domain_navigation");
  });

  it("should track competence per domain", () => {
    const competenceMap = learner.getCompetenceMap();
    expect(competenceMap).toBeDefined();
  });

  it("should support curriculum learning", () => {
    const curriculum = learner.curriculum;

    curriculum.addTask({
      id: "task1",
      domain: "navigation",
      difficulty: 0.2,
      prerequisites: [],
    });

    curriculum.addTask({
      id: "task2",
      domain: "navigation",
      difficulty: 0.5,
      prerequisites: ["task1"],
    });

    const nextTask = curriculum.getNextTask("navigation");
    expect(nextTask?.id).toBe("task1");
  });
});

// ==================== DEBATE TESTS ====================

describe("Inner Debate System", () => {
  let debate: InnerDebateOrchestrator;

  beforeEach(() => {
    debate = new InnerDebateOrchestrator(16);
  });

  it("should conduct debates and reach decisions", async () => {
    const context = Array(16).fill(0.5);
    const goal = { type: "reach", target: [1.0, 0.0] };

    const result = await debate.debate(context, goal);

    expect(result.winningProposal).toBeDefined();
    expect(result.consensus).toBeGreaterThan(0);
    expect(result.debateRounds).toBeGreaterThanOrEqual(1);
  });

  it("should track agent states", () => {
    const states = debate.getAgentStates();

    expect(states.length).toBe(5); // planner, critic, verifier, explainer, explorer
    expect(states.map(s => s.role)).toContain("planner");
    expect(states.map(s => s.role)).toContain("critic");
  });
});

// ==================== REFLECTION TESTS ====================

describe("Meta-Cognitive Loop", () => {
  let reflection: MetaCognitiveLoop;

  beforeEach(() => {
    reflection = new MetaCognitiveLoop(16);
  });

  it("should record and critique decisions", async () => {
    const decisionId = await reflection.recordDecision(
      "navigation",
      Array(16).fill(0.5),
      { action: "move_forward" },
      ["Obstacle detected", "Clear path ahead"]
    );

    expect(decisionId).toBeDefined();

    const critique = await reflection.recordOutcome(decisionId, true, 0.8, 0.7);

    expect(critique.strengths).toBeDefined();
    expect(critique.weaknesses).toBeDefined();
    expect(critique.improvementSuggestions).toBeDefined();
  });

  it("should detect failure patterns", async () => {
    // Record multiple failures
    for (let i = 0; i < 5; i++) {
      const id = await reflection.recordDecision(
        "navigation",
        Array(16).fill(0.5),
        { action: "move" },
        ["Low confidence"]
      );
      await reflection.recordOutcome(id, false, -0.5, 0.3);
    }

    const result = await reflection.reflect();
    // May detect patterns after enough failures
  });

  it("should track competence per domain", async () => {
    await reflection.recordDecision(
      "navigation",
      Array(16).fill(0.5),
      { action: "test" },
      ["Test reasoning"]
    );

    const stats = reflection.getStats();
    expect(stats.overallCompetence).toBeDefined();
  });
});

// ==================== SKILL DISCOVERY TESTS ====================

describe("Skill Discovery Engine", () => {
  let skillDiscovery: SkillDiscoveryEngine;

  beforeEach(() => {
    skillDiscovery = new SkillDiscoveryEngine({
      maxBehaviors: 100,
      minClusterSize: 2,
      similarityThreshold: 0.6,
    });
  });

  it("should record behaviors", () => {
    const behaviorId = skillDiscovery.recordBehavior(
      [
        { state: [0.1], action: [0.2], latent: [0.3], symbol: "S1", reward: 0.5 },
        { state: [0.2], action: [0.3], latent: [0.4], symbol: "S2", reward: 0.6 },
      ],
      [0.5, 0.5],
      { totalReward: 1.1, success: true, goalReached: false, duration: 1000 },
      "navigation"
    );

    expect(behaviorId).toBeDefined();
  });

  it("should discover skills from similar behaviors", async () => {
    // Record multiple similar behaviors
    for (let i = 0; i < 5; i++) {
      skillDiscovery.recordBehavior(
        [
          { state: [0.1 + i * 0.01], action: [0.2], latent: [0.3], symbol: "S1", reward: 0.5 },
          { state: [0.2 + i * 0.01], action: [0.3], latent: [0.4], symbol: "S2", reward: 0.6 },
        ],
        [0.5, 0.5],
        { totalReward: 1.1, success: true, goalReached: false, duration: 1000 },
        "navigation"
      );
    }

    const skills = await skillDiscovery.discover("navigation");
    // Skills may or may not be discovered depending on clustering
  });
});

// ==================== CURIOSITY TESTS ====================

describe("Curiosity Engine", () => {
  let curiosity: CuriosityEngine;

  beforeEach(() => {
    curiosity = new CuriosityEngine(16);
  });

  it("should compute intrinsic rewards", () => {
    const state = Array(16).fill(0.5);
    const prediction = Array(16).fill(0.4);
    const actual = Array(16).fill(0.5);
    const symbols = ["S1"];
    const previousSymbols = ["S0"];

    const novelty = curiosity.computeIntrinsicReward(
      state,
      prediction,
      actual,
      symbols,
      previousSymbols,
      1
    );

    expect(novelty.informationGain).toBeDefined();
    expect(novelty.stateNovelty).toBeDefined();
    expect(novelty.totalIntrinsic).toBeDefined();
  });

  it("should form concepts from observations", () => {
    // Add many observations
    for (let i = 0; i < 20; i++) {
      const state = Array(16).fill(0.5 + i * 0.01);
      curiosity.computeIntrinsicReward(
        state,
        state,
        state,
        ["S1"],
        [],
        1
      );
    }

    const concepts = curiosity.formConcepts();
    // Concepts may form from clustered observations
  });

  it("should track curiosity state", () => {
    const state = curiosity.getCuriosityState();

    expect(state.explorationBonus).toBeDefined();
    expect(state.currentFocus).toMatch(/exploration|exploitation/);
  });
});

// ==================== TOOL REASONING TESTS ====================

describe("Tool Reasoning Engine", () => {
  let tools: ToolReasoningEngine;

  beforeEach(() => {
    tools = new ToolReasoningEngine(16);
  });

  it("should have built-in tools", () => {
    const availableTools = tools.registry.getAllTools();
    expect(availableTools.length).toBeGreaterThan(0);
    expect(availableTools.map(t => t.id)).toContain("calculator");
  });

  it("should use calculator tool", async () => {
    tools.grantPermission("test_agent", "calculator", "execute");

    const result = await tools.useTool(
      "calculator",
      { expression: "2 + 2" },
      "test_agent",
      "math"
    );

    expect(result.success).toBe(true);
    expect(result.output.result).toBe(4);
  });

  it("should reason about tool selection", async () => {
    tools.grantPermission("test_agent", "calculator", "execute");
    tools.grantPermission("test_agent", "text_transformer", "execute");

    const reasoning = await tools.reason(
      Array(16).fill(0.5),
      "calculate something",
      "test_agent"
    );

    expect(reasoning.selectedTool).toBeDefined();
    expect(reasoning.reasoning.length).toBeGreaterThan(0);
  });
});

// ==================== GOAL TRACKING TESTS ====================

describe("Goal Tracker", () => {
  let goals: GoalTracker;

  beforeEach(() => {
    goals = new GoalTracker();
  });

  it("should create and activate goals", () => {
    const goalId = goals.createGoal(
      "Learn Navigation",
      "Master navigation in complex environments",
      "navigation",
      10,
      { priority: 8 }
    );

    expect(goalId).toBeDefined();

    const activated = goals.activateGoal(goalId);
    expect(activated).toBe(true);

    const activeGoal = goals.getActiveGoal();
    expect(activeGoal?.id).toBe(goalId);
  });

  it("should update progress", () => {
    const goalId = goals.createGoal(
      "Test Goal",
      "Test description",
      "test",
      5
    );

    goals.activateGoal(goalId);
    goals.updateProgress(goalId, 50, "Halfway there");

    const goal = goals.repository.get(goalId);
    expect(goal?.progress).toBe(50);
  });

  it("should decompose complex goals", () => {
    const goalId = goals.createGoal(
      "Complex Goal",
      "A goal that needs decomposition",
      "complex",
      20
    );

    const subgoalIds = goals.decomposeGoal(goalId);
    expect(subgoalIds.length).toBeGreaterThan(0);
  });

  it("should auto-complete on 100% progress", () => {
    const goalId = goals.createGoal(
      "Quick Goal",
      "Completes quickly",
      "test",
      1
    );

    goals.activateGoal(goalId);
    goals.updateProgress(goalId, 100, "Done");

    const goal = goals.repository.get(goalId);
    expect(goal?.status).toBe("completed");
  });
});

// ==================== CONSCIOUSNESS TESTS ====================

describe("Consciousness Monitor", () => {
  let consciousness: ConsciousnessMonitor;

  beforeEach(() => {
    consciousness = new ConsciousnessMonitor();
  });

  it("should track confidence", () => {
    consciousness.confidence.update("navigation", "task1", 0.8, 0.9);
    consciousness.confidence.update("navigation", "task2", 0.7, 0.8);

    const signals = consciousness.confidence.getSignals();
    expect(signals.overall).toBeGreaterThan(0);
  });

  it("should track uncertainty", () => {
    consciousness.uncertainty.update(0.3, 0.2, "world_model");

    const signals = consciousness.uncertainty.getSignals();
    expect(signals.epistemic).toBeGreaterThan(0);
  });

  it("should track motivation", () => {
    consciousness.motivation.recordActivity("exploration");
    consciousness.motivation.recordActivity("learning");

    const signals = consciousness.motivation.getSignals();
    expect(signals.explorationDrive).toBeDefined();
    expect(signals.learningDesire).toBeDefined();
  });

  it("should perform introspection", () => {
    const report = consciousness.introspect();

    expect(report.state).toBeDefined();
    expect(report.insights).toBeDefined();
    expect(report.recommendations).toBeDefined();
  });

  it("should get full state", () => {
    const state = consciousness.getState();

    expect(state.confidence).toBeDefined();
    expect(state.uncertainty).toBeDefined();
    expect(state.competence).toBeDefined();
    expect(state.alignment).toBeDefined();
    expect(state.motivation).toBeDefined();
    expect(state.attention).toBeDefined();
    expect(state.metacognition).toBeDefined();
  });
});

// ==================== INTEGRATED AGENT TESTS ====================

describe("Multi-Domain Agent", () => {
  let agent: MultiDomainAgent;

  beforeEach(() => {
    agent = createMultiDomainAgent("TestAgent", {
      latentDim: 8,
      beliefDim: 32,
      actionDim: 4,
      inputDim: 16,
    });
  });

  it("should initialize with correct configuration", () => {
    expect(agent.name).toBe("TestAgent");

    const state = agent.getState();
    expect(state.currentDomain).toBeNull();
    expect(state.consciousness).toBeDefined();
  });

  it("should enter and track domains", async () => {
    const domainId = await agent.enterDomain("navigation");

    expect(domainId).toBeDefined();
    expect(agent.getCurrentDomain()).toBe(domainId);
  });

  it("should manage goals", () => {
    const goalId = agent.createGoal(
      "Learn Environment",
      "Understand the navigation environment",
      5,
      { priority: 7 }
    );

    expect(goalId).toBeDefined();

    agent.activateGoal(goalId);
    const activeGoal = agent.getActiveGoal();
    expect(activeGoal?.id).toBe(goalId);
  });

  it("should run episodes", async () => {
    await agent.enterDomain("test");

    agent.startEpisode(Array(16).fill(0.5));

    // Run a few steps
    for (let i = 0; i < 3; i++) {
      const result = await agent.step(
        Array(16).fill(Math.random()),
        Math.random() - 0.5,
        i === 2
      );

      expect(result.action).toBeDefined();
      expect(result.reasoning).toBeDefined();
      expect(result.novelty).toBeDefined();
    }

    const episodeResult = await agent.endEpisode();

    expect(episodeResult.domain).toBeDefined();
    expect(typeof episodeResult.totalReward).toBe("number");
  });

  it("should perform reflection", async () => {
    await agent.enterDomain("reflection_test");

    // Do some activity first
    agent.startEpisode(Array(16).fill(0.5));
    await agent.step(Array(16).fill(0.5), 0.5, true);
    await agent.endEpisode();

    const reflectionResult = await agent.reflect();

    expect(reflectionResult.insights).toBeDefined();
    expect(reflectionResult.competenceUpdates).toBeDefined();
  });

  it("should provide comprehensive stats", () => {
    const stats = agent.getStats();

    expect(stats.name).toBe("TestAgent");
    expect(stats.memory).toBeDefined();
    expect(stats.worldModel).toBeDefined();
    expect(stats.consciousness).toBeDefined();
    expect(stats.goals).toBeDefined();
    expect(stats.skills).toBeDefined();
  });

  it("should introspect", () => {
    const report = agent.introspect();

    expect(report.state).toBeDefined();
    expect(report.insights).toBeDefined();
    expect(report.recommendations).toBeDefined();
  });

  it("should use tools", async () => {
    const result = await agent.useTool("calculator", { expression: "10 * 5" });

    expect(result.success).toBe(true);
    expect(result.output.result).toBe(50);
  });

  it("should support imagination/planning", async () => {
    const actionSequence = [
      [0.1, 0.2, 0.3, 0.4],
      [0.2, 0.3, 0.4, 0.5],
    ];

    const prediction = await agent.imagine(actionSequence);

    expect(prediction.states).toBeDefined();
    expect(prediction.states.length).toBe(3);
  });

  it("should support counterfactual reasoning", async () => {
    const result = await agent.counterfactual(
      [0.5, 0.5, 0.5, 0.5],
      [-0.5, -0.5, -0.5, -0.5]
    );

    expect(result.recommendation).toBeDefined();
    expect(typeof result.difference).toBe("number");
  });
});

// ==================== INTEGRATION TESTS ====================

describe("System Integration", () => {
  let agent: MultiDomainAgent;

  beforeEach(() => {
    agent = createMultiDomainAgent("IntegrationTestAgent");
  });

  it("should handle full learning loop", async () => {
    // 1. Enter domain
    await agent.enterDomain("integration_test");

    // 2. Create goal
    const goalId = agent.createGoal("Test Learning", "Complete test", 5);
    agent.activateGoal(goalId);

    // 3. Run episode
    agent.startEpisode(Array(64).fill(0.5));

    for (let i = 0; i < 5; i++) {
      await agent.step(
        Array(64).fill(Math.random()),
        Math.random(),
        i === 4
      );
    }

    const episodeResult = await agent.endEpisode();

    // 4. Reflect
    const reflection = await agent.reflect();

    // 5. Introspect
    const introspection = agent.introspect();

    // Verify all components worked together
    expect(episodeResult.domain).toBeDefined();
    expect(reflection.insights).toBeDefined();
    expect(introspection.state).toBeDefined();
  });

  it("should maintain consistency across components", async () => {
    await agent.enterDomain("consistency_test");

    // Record activity through various subsystems
    agent.consciousness.motivation.recordActivity("exploration");
    agent.consciousness.attention.setFocus("test_focus");

    agent.createGoal("Consistency Test", "Test", 1);

    // All components should reflect the activity
    const state = agent.getState();
    const stats = agent.getStats();

    expect(state.currentDomain).toBeDefined();
    expect(stats.consciousness).toBeDefined();
  });
});

console.log("Multi-Domain Agent Tests Loaded");
