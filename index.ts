import { AMKernel } from "./core/am_kernel";
import { ExampleEnv } from "./envs/env_example";
import { launchAPI } from "./core/api";

const env = new ExampleEnv();
export const core = new AMKernel(env, 4, 2);

(async () => {
  setInterval(() => core.runEpisode("task0"), 500);
})();
launchAPI(core);
