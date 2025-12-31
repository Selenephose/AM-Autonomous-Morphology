// core/index.ts

import { AM_Organism } from "./organism"
import { AM_Cortex } from "./cortex"
import * as math from "mathjs"
import express, { Request, Response } from "express"

// ================= Dummy Environment =================
class DummyEnv {
  async reset() {
    return { state: [0, 0, 0, 0], reward: 0, done: false }
  }
  async step(_: number[]) {
    const s = [Math.random(), Math.random(), Math.random(), Math.random()]
    return { state: s, reward: math.sum(s) as number, done: Math.random() > 0.95 }
  }
}

// ================= Boot Organism =================
const env = new DummyEnv()
const organism = new AM_Organism(4, 2)
const cortex = new AM_Cortex(organism)

// ================= Autonomous Symbol Genesis =================
setInterval(async () => {
  const obs = await env.reset()
  const z = organism.encode(obs.state)
  const sym = "S_" + Math.floor(math.sum(z) * 100)
  cortex.observe(sym, "task0", "", z)
}, 1000)

// ================= REST API =================
const app = express()

app.get("/metrics", (req: Request, res: Response) => {
  res.json(cortex.metrics())
})

app.get("/hierarchy", (req: Request, res: Response) => {
  res.json(cortex.hierarchy())
})

app.listen(5030, () => {
  console.log("🧠 AM Cortex online at http://localhost:5030")
})
