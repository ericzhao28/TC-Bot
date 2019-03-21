#!/usr/bin/env bash

# simDQN="./deep_dialog/checkpoints/rl_agent/agt_9_400_500_0.87400.p"
simDQN="./deep_dialog/checkpoints/rl_agent/agt_9_400_400_0.91000.p"
simDagger="./deep_dialog/checkpoints/rl_agent/agt_10_200_1000_0.86000.p"
realDQN="./deep_dialog/checkpoints/rl_agent/agt_9_800_800_0.43000.p"
v2DQN="./deep_dialog/checkpoints/rl_agent/agt_9_600_1000_0.46800.p"
newSimDQN=""
newSimDagger=""

# Train sim raw-level DQN model
# python run.py --agt 9 --usr 1 --max_turn 20 --dqn_hidden_size 80 --kb_path ./deep_dialog/data/movie_kb.1k.p --experience_replay_pool_size 1000 --episodes 1001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 1 --warm_start_epochs 120 --train 1 --epsilon 0.01

# Train sim raw-level Dagger model
# python run.py --agt 10 --usr 1 --max_turn 20 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --dagger_hidden_size 80 --episodes 1001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --expert_model_path $simDQN

# Resume train DQN model in nl-level
# python run.py --agt 9 --usr 1 --max_turn 20 --dqn_hidden_size 80 --kb_path ./deep_dialog/data/movie_kb.1k.p --experience_replay_pool_size 1000 --episodes 2001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --trained_model_path $simDQN

# Resume train Dagger model in nl-level
# python run.py --agt 10 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --dagger_hidden_size 80 --episodes 1001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 32 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --expert_model_path $realDQN --trained_model_path $simDagger

# Resume train expert-guided model in nl-level
# python run.py --agt 11 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 1001 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --train 1 --expert_model_path $realDQN --trained_model_path $simDQN

# Evaluate direct NL-level DQN model with real user
# python run.py --agt 9 --usr 1 --max_turn 20 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --trained_model_path $simDQN

# Evaluate direct NL-level Dagger model with real user
# python run.py --agt 10 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --episodes 10 --simulation_epoch_size 100 --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --dagger_hidden_size 80 --trained_model_path $newSimDagger

# Evaluate new model with real user
# python run.py --agt 9 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 3 --act_level 1 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 0 --trained_model_path $newExpertGuidedDagger