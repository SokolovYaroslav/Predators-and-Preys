import json

from tqdm.auto import tqdm

from boosted_baseline.agents import PredatorAgent, PreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


def gather():
    env = PredatorsAndPreysEnv(render=False)
    predator_agent = PredatorAgent()
    prey_agent = PreyAgent()
    with open("states.jsonl", "w") as f:
        bar = tqdm()
        try:
            while True:
                state_dict = env.reset()
                while True:
                    pred_act, prey_act = predator_agent.act(state_dict), prey_agent.act(state_dict)
                    state_dict, _, done = env.step(pred_act, prey_act)
                    f.write(json.dumps({"state": state_dict, "pred_act": pred_act, "prey_act": prey_act}) + "\n")
                    bar.update()
                    if done:
                        break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    gather()
