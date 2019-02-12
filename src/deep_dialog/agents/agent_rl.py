"""
Created on Feb, 11, 2019

Template for RL agents


@author: ericzhao28
"""

import random, copy
import numpy as np

from deep_dialog import dialog_config

from agent import Agent


class AgentRL(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.max_turn = params["max_turn"] + 4
        self.state_dimension = (
            2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        )

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0

        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0

        # self.request_set = dialog_config.movie_request_slots #['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def initialize_config(self, req_set, inf_set):
        """ Initialize request_set and inform_set """

        self.request_set = req_set
        self.inform_set = inf_set
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0

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
        if (not self.evaluation_mode) and (random.random() < self.epsilon):
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                return self.rule_request_inform_policy()
                # return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_mode=True)

    def rule_policy(self):
        """ Rule Policy """

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response["diaact"] = "request"
            act_slot_response["inform_slots"] = {}
            act_slot_response["request_slots"] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {
                "diaact": "inform",
                "inform_slots": {"taskcomplete": "PLACEHOLDER"},
                "request_slots": {},
            }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {
                "diaact": "thanks",
                "inform_slots": {},
                "request_slots": {},
            }

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print
        act_slot_response
        raise Exception("action index not found")
        return None
