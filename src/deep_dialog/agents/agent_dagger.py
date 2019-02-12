"""
Created on Feb, 11, 2019

An Dagger Agent
- An Dagger

@author: ericzhao28
"""


import random, copy, json
import cPickle as pickle
import numpy as np

from deep_dialog import dialog_config

from agent import Agent
from deep_dialog.neural import Dagger


class AgentDagger(Agent):
    def __init__(
        self, expert, movie_dict=None, act_set=None, slot_set=None, params=None
    ):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params["epsilon"]
        self.agent_run_mode = params["agent_run_mode"]
        self.agent_act_level = params["agent_act_level"]
        self.dagger_X = []  # dagger data pool <s_t>
        self.dagger_Y = []  # dagger data pool <a_t>

        self.hidden_size = params.get("dagger_hidden_size", 60)
        self.gamma = params.get("gamma", 0.9)
        self.predict_mode = params.get("predict_mode", False)

        self.max_turn = params["max_turn"] + 4
        self.state_dimension = (
            2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        )

        self.dagger = Dagger(self.state_dimension, self.hidden_size, self.num_actions)

        self.warm_start = params.get("warm_start", 0)

        self.expert = expert

        self.cur_bellman_err = 0

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = [
            "moviename",
            "starttime",
            "city",
            "date",
            "theater",
            "numberofpeople",
        ]

    def state_to_action(self, state):
        """ DQN: Input state, output action """

        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {"act_slot_response": act_slot_response, "act_slot_value_response": None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state["user_action"]
        current_slots = state["current_slots"]
        kb_results_dict = state["kb_results_dict"]
        agent_last = state["agent_action"]

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action["diaact"]]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action["inform_slots"].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action["request_slots"].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots["inform_slots"]:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last["diaact"]]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last["inform_slots"].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last["request_slots"].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + state["turn"] / 10.0

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state["turn"]] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = (
            np.zeros((1, self.slot_cardinality + 1))
            + kb_results_dict["matching_all_constraints"] / 100.0
        )
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.0

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict["matching_all_constraints"] > 0.0
        )
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(
                    kb_results_dict[slot] > 0.0
                )

        self.final_representation = np.hstack(
            [
                user_act_rep,
                user_inform_slots_rep,
                user_request_slots_rep,
                agent_act_rep,
                agent_inform_slots_rep,
                agent_request_slots_rep,
                current_slots_rep,
                turn_rep,
                turn_onehot_rep,
                kb_binary_rep,
                kb_count_rep,
            ]
        )
        return self.final_representation

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                return self.expert.run_policy(representation)
            return self.dagger.predict(representation)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_dagger_tuple(self, s_t):
        """ Register feedback from the environment, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        a_t = self.expert.run_policy(state_t_rep)

        # if not (self.predict_mode == False):
        self.dagger_X.append(state_t_rep[0])
        self.dagger_Y.append(a_t)

    def train(self, batch_size=1, _=None):
        """ Train DQN with experience replay """
        print (len(self.dagger_X))
        self.dagger.train(self.dagger_X, self.dagger_Y, batch_size)
