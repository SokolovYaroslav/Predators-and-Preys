from predators_and_preys_env.env import PredatorsAndPreysEnv
from test_submission.submission import PredatorAgent, PreyAgent


def test_main():
    env = PredatorsAndPreysEnv(render=True)
    predator_agent = PredatorAgent()
    prey_agent = PreyAgent()

    while True:
        state_dict = env.reset()
        while True:
            pred_act, prey_act = predator_agent.act(state_dict), prey_agent.act(state_dict)
            state_dict, _, done = env.step(pred_act, prey_act)
            if done:
                break


if __name__ == "__main__":
    test_main()
