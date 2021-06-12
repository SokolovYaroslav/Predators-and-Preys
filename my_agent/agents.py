import numpy as np

from my_agent.states import GameState, EntityState
from predators_and_preys_env.agent import PredatorAgent, PreyAgent


class ChasingPredatorAgent(PredatorAgent):
    def act(self, state_dict):
        state_dict = GameState(state_dict)
        action = []
        for predator in state_dict.predators:
            closest_prey = None
            for prey in state_dict.preys:
                if not prey.is_alive:
                    continue
                if closest_prey is None:
                    closest_prey = prey
                else:
                    if closest_prey.dist(predator) > prey.dist(predator):
                        closest_prey = prey
            if closest_prey is None:
                action.append(0.0)
            else:
                action.append(predator.direction(closest_prey))
        return action


class FleeingPreyAgent(PreyAgent):
    def act(self, state_dict):
        state_dict = GameState(state_dict)
        action = []
        for prey in state_dict.preys:
            closest_predator = None
            for predator in state_dict.predators:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if closest_predator.dist(prey) > prey.dist(predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.0)
            else:
                action.append(closest_predator.direction(prey))
        return action
