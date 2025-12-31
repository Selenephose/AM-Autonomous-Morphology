export type EnvObs = { state: number[]; reward: number; done: boolean };
export interface Env {
  reset(): Promise<EnvObs>;
  step(a: number[]): Promise<EnvObs>;
}
