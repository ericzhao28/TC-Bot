#!/usr/bin/env bash

# CMD Agent
# python run.py --agt 0 --usr 1 --max_turn 40 --episodes 150 --kb_path ./deep_dialog/data/movie_kb.1k.p --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --intent_err_prob 0.00 --slot_err_prob 0.00 --episodes 500 --act_level 0 --run_mode 0  --cmd_input_mode 0

# Continue to train old DQN model
# python run.py --agt 9 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 2000 --simulation_epoch_size 10 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --train 1 --trained_model_path $simDQN


# Eval DQN pretrained model
# python run.py --agt 9 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --trained_model_path ./deep_dialog/checkpoints/rl_agent/agt_9_650_1580_0.70000.p

# Play with yourself
# python run.py --agt 0 --usr 0 --max_turn 40 --episodes 5 --kb_path ./deep_dialog/data/movie_kb.1k.p --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --intent_err_prob 0.00 --slot_err_prob 0.00 --episodes 500 --act_level 0 --run_mode 0  --cmd_input_mode 0