/**
 * AM Autonomous Morphology - Public API
 *
 * Re-exports all components from the core module
 */

// Re-export everything from core
export * from "../core";

// Explicit exports for main entry points
export { MultiDomainAgent, createMultiDomainAgent } from "../core/multi_domain_agent";
export { AM_Organism } from "../core/organism";
export { AM_Cortex } from "../core/cortex";
