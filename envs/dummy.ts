import * as math from "mathjs";

export class DummyEnv {
  async reset(){ return {state:[0,0,0,0], reward:0, done:false}; }
  async step(_:number[]){
    const s=[Math.random(),Math.random(),Math.random(),Math.random()];
    return {state:s, reward:math.sum(s) as number, done:Math.random()>0.95};
  }
}
