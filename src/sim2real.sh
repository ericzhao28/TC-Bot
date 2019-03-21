#!/usr/bin/env bash

simDQN="./deep_dialog/checkpoints/rl_agent/agt_9_400_400_0.91000.p"
simDagger="./deep_dialog/checkpoints/rl_agent/agt_10_200_1000_0.86000.p"
realSimDQN=""
realSimDagger=""

# Train sim DQN model
# python run.py --agt 9 --usr 1 --max_turn 20 --dqn_hidden_size 80 --kb_path ./deep_dialog/data/movie_kb.1k.p --experience_replay_pool_size 1000 --episodes 1001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 1 --warm_start_epochs 120 --train 1 --epsilon 0.01

# Train sim Dagger model
# python run.py --agt 10 --usr 1 --max_turn 20 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --dagger_hidden_size 80 --episodes 1001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --expert_model_path $simDQN

# Resume train DQN model in real world
# python run.py --agt 9 --usr 0 --max_turn 20 --dqn_hidden_size 80 --kb_path ./deep_dialog/data/movie_kb.1k.p --experience_replay_pool_size 1000 --episodes 50 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --trained_model_path $simDQN --epsilon 0.01

# Resume train Dagger model in real world
# python run.py --agt 10 --usr 0 --max_turn 20 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --dagger_hidden_size 80 --episodes 1001 --simulation_epoch_size 10 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 0 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --trained_model_path $simDagger

# Train new model

# Evaluate direct NL-level DQN model with real user
python run.py --agt 9 --usr 0 --max_turn 20 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --episodes 10 --simulation_epoch_size 10 --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --trained_model_path $simDQN

# Evaluate direct NL-level Dagger model with real user
# python run.py --agt 10 --usr 0 --max_turn 20 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --episodes 10 --simulation_epoch_size 10 --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --dagger_hidden_size 80 --trained_model_path $realSimDagger

# Evaluate new model with real user