"""
Created on Jun 18, 2016

A focused DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN
- Use expert model as behavior policy.

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: ericzhao28
"""


import random
from agent_dqn import AgentDQN


class AgentFocusedDQN(AgentDQN):
    def __init__(self, expert, movie_dict=None, act_set=None, slot_set=None, params=None):
        super(AgentFocusedDQN, self).__init__(movie_dict, act_set, slot_set, params)
        self.expert = expert

    def run_policy(self, representation):
        """ epsilon-greedy policy """
        if (not self.evaluation_mode) and (random.random() < self.epsilon):
            return random.randint(0, self.num_actions - 1)
        elif self.evaluation_mode:
            return self.dqn.predict(representation, {}, predict_mode=True)
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