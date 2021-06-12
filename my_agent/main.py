from my_agent.agents import ChasingPredatorAgentResidual, FleeingPreyAgentResidual
from predators_and_preys_env.env import PredatorsAndPreysEnv


def main():
    env = PredatorsAndPreysEnv(render=True)
    predator_agent = ChasingPredatorAgentResidual()
    prey_agent = FleeingPreyAgentResidual()

    done = True
    step_count = 0
    state_dict = None
    while True:
        if done:
            state_dict = env.reset()
            step_count = 0

        state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
        step_count += 1


if __name__ == "__main__":
    main()
