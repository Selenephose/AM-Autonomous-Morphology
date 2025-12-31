import express from "express";
export function launchAPI(core) {
  const app = express();
  app.get("/am/metrics",(req,res)=>res.json(core.metrics.snapshot()));
  app.get("/am/hierarchy",(req,res)=>res.json(core.graph.hierarchy()));
  app.get("/am/logs",(req,res)=>res.json(core.logs.slice(-5)));
  app.listen(5030,()=>console.log("AM Kernel listening on :5030"));
}
