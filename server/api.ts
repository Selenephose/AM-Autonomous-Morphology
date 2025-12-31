import express from "express"
import { DummyEnv } from "../env/dummy"
import { AMKernel } from "../core/kernel"
import { AM_Cortex } from "../core/cortex"

const env = new DummyEnv()
const kernel = new AMKernel(env, 4, 2)
const cortex = new AM_Cortex(kernel)

setInterval(() => kernel.runEpisode("task0"), 1000)

const app = express()
app.get("/metrics", (_,res)=>res.json(cortex.getPhaseMetrics()))
app.get("/hierarchy", (_,res)=>res.json(cortex.getHierarchy()))
app.listen(5030, ()=>console.log("AM listening on :5030"))
