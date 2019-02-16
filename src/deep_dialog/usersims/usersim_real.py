"""
Created on May 14, 2016

a rule-based user simulator

-- user_goals_first_turn_template.revised.v1.p: all goals
-- user_goals_first_turn_template.part.movie.v1.p: moviename in goal.inform_slots
-- user_goals_first_turn_template.part.nomovie.v1.p: no moviename in goal.inform_slots

@author: xiul, t-zalipt
"""

from .usersim_rule import RuleSimulator
import argparse, json, random, copy

from deep_dialog import dialog_config


class RealUser(RuleSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def initialize_episode(self):
        """ Initialize a new episode (dialog) 
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state["history_slots"] = {}
        self.state["inform_slots"] = {}
        self.state["request_slots"] = {}
        self.state["rest_slots"] = []
        self.state["turn"] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        # self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.goal["request_slots"]["ticket"] = "UNK"
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        """ Debug: build a fake goal mannually """
        # self.debug_falk_goal()

        # sample first action
        print json.dumps(self.goal, indent=2)
        user_action = self._sample_action()
        assert self.episode_over != 1, " but we just started"
        return user_action

    def _sample_action(self):
        user_input = raw_input("Say next: ")

        sample_action = {}
        sample_action["turn"] = self.state["turn"]
        sample_action["nl"] = user_input

        user_nlu_res = self.nlu_model.generate_dia_act(user_input)  # NLU
        assert user_nlu_res != None

        sample_action.update(user_nlu_res)
        print(sample_action.keys())
        return sample_action

    def next(self, system_action):
        """ Generate next User Action based on last System Action """

        self.state["turn"] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        sys_act = system_action["diaact"]

        if self.max_turn > 0 and self.state["turn"] > self.max_turn:
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True

        while self.dialog_status == dialog_config.NO_OUTCOME_YET:
            user_input = raw_input("Episode result (0-continue, 1-success, 2-fail): ")
            try:
                user_input = int(user_input)
                assert(user_input in [0,1,2])
            except:
                continue
            if user_input == 1:
                self.episode_over = True
                self.dialog_status = dialog_config.SUCCESS_DIALOG
            elif user_input == 2:
                self.episode_over = True
                self.dialog_status = dialog_config.FAILED_DIALOG
            break

        response_action = self._sample_action()
        return response_action, self.episode_over, self.dialog_status


def main(params):
    user_sim = RealUser()
    user_sim.initialize_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print("User Simulator Parameters:")
    print(json.dumps(params, indent=2))

    main(params)
