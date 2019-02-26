"""
Created on Feb, 11, 2019

An Dagger Agent
- An Dagger

@author: ericzhao28
"""


import random, copy, pickle


from agent_rl import AgentRL
from agent_cmd import AgentCmd
from deep_dialog.neural import Dagger


class AgentDagger(AgentRL):
    def __init__(
        self, expert, movie_dict=None, act_set=None, slot_set=None, params=None
    ):
        super(AgentDagger, self).__init__(movie_dict, act_set, slot_set, params)

        self.epsilon = params["epsilon"]
        self.agent_run_mode = params["agent_run_mode"]
        self.agent_act_level = params["agent_act_level"]
        self.dagger_X = []  # dagger data pool <s_t>
        self.dagger_Y = []  # dagger data pool <a_t>

        self.hidden_size = params.get("dagger_hidden_size", 60)
        self.gamma = params.get("gamma", 0.9)
        self.predict_mode = params.get("predict_mode", False)

        self.dagger = Dagger(self.state_dimension, self.hidden_size, self.num_actions)

        self.warm_start = params.get("warm_start", 0)

        self.expert = expert
        self.cache = {}

        self.cur_bellman_err = 0

    def run_policy(self, representation):
        """ epsilon-greedy policy """
        if (not self.evaluation_mode) and (random.random() < self.epsilon):
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                assert(not isinstance(self.expert, AgentCmd))
                return self.expert.run_policy(representation)
            return self.dagger.predict(representation)

    def register_step(self, s_t, *args):
        """ Register feedback from the environment, to be stored as future training data """
        if self.evaluation_mode:  # Training Mode
            return
        state_t_rep = self.prepare_state_representation(s_t)
        if isinstance(self.expert, AgentCmd):
            if str(state_t_rep) not in self.cache:
                a_t = self.expert.run_policy(s_t)
                self.cache[str(state_t_rep)] = a_t
            else:
                a_t = self.cache[str(state_t_rep)]
        else:
            a_t = self.expert.run_policy(state_t_rep)
        self.dagger_X.append(state_t_rep[0])
        self.dagger_Y.append(a_t)

    def train(self, batch_size=1, _=None):
        """ Train DQN with experience replay """
        self.dagger.train(self.dagger_X, self.dagger_Y, batch_size)

    def load_trained_dagger(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, "rb"))
        model = trained_file["model"]

        self.dagger.model = copy.deepcopy(model)
