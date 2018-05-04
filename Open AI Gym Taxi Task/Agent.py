import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        

    def select_action(self, state, episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        eps = 1/((0.9 * episode) + 1)
        policy_s = np.ones(self.nA) * eps / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - eps + (eps / self.nA)
        
        return np.random.choice(np.arange(self.nA), p = policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Q[state][action] = update_Q(Q[state][action], np.max(Q[next_state]), reward, alpha, gamma)
        self.Q[state][action] = self.Q[state][action] + (0.1 * (reward + 0.99 * (np.max(self.Q[next_state]))) - self.Q[state][action])
        