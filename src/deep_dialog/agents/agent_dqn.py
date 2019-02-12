"""
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: xiul, ericzhao28
"""


import random, copy, json
import cPickle as pickle

from agent_rl import AgentRL
from deep_dialog.neural import DQN


class AgentDQN(AgentRL):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        super(AgentDQN, self).__init__(movie_dict, act_set, slot_set, params)

        self.epsilon = params["epsilon"]
        self.agent_run_mode = params["agent_run_mode"]
        self.agent_act_level = params["agent_act_level"]
        self.experience_replay_pool = (
            []
        )  # experience replay pool <s_t, a_t, r_t, s_t+1>

        self.experience_replay_pool_size = params.get(
            "experience_replay_pool_size", 1000
        )
        self.hidden_size = params.get("dqn_hidden_size", 60)
        self.gamma = params.get("gamma", 0.9)
        self.warm_start = params.get("warm_start", 0)

        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.clone_dqn = copy.deepcopy(self.dqn)

        self.cur_bellman_err = 0

    def run_policy(self, representation):
        """ epsilon-greedy policy """
        if (not self.evaluation_mode) and (random.random() < self.epsilon):
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_mode=True)

    def register_step(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """
        if self.evaluation_mode:  # Training Mode
            return
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (
            state_t_rep,
            action_t,
            reward_t,
            state_tplus1_rep,
            episode_over,
        )
        self.experience_replay_pool.append(training_example)

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """
        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            for iter in range(len(self.experience_replay_pool) / (batch_size)):
                batch = [
                    random.choice(self.experience_replay_pool)
                    for i in xrange(batch_size)
                ]
                batch_struct = self.dqn.singleBatch(
                    batch, {"gamma": self.gamma}, self.clone_dqn
                )
                self.cur_bellman_err += batch_struct["cost"]["total_cost"]

            print (
                "cur bellman err %.4f, experience replay pool %s"
                % (
                    float(self.cur_bellman_err) / len(self.experience_replay_pool),
                    len(self.experience_replay_pool),
                )
            )

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print "saved model in %s" % (path,)
        except Exception, e:
            print "Error: Writing model fails: %s" % (path,)
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, "rb"))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, "rb"))
        model = trained_file["model"]

        self.dqn.model = copy.deepcopy(model)
        self.clone_dqn = copy.deepcopy(self.dqn)
