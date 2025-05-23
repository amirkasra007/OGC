DEFAULTVALUE=4
DEFAULTSEED=2
device="${1:-$DEFAULTVALUE}"
seed="${2:-$DEFAULTSEED}"
CUDA_VISIBLE_DEVICES=${device} XLA_PYTHON_CLIENT_MEM_FRACTION=.40 LD_LIBRARY_PATH="" nice -n 5 python3 -m minimax.train \
--wandb_mode=online \
--wandb_project=overcooked-minimax-jax \
--wandb_entity=${WANDB_ENTITY} \
--seed=${seed} \
--agent_rl_algo=ppo \
--n_total_updates=30000 \
--train_runner=paired \
--n_devices=1 \
--student_model_name=default_student_actor_cnn \
--student_critic_model_name=default_student_critic_cnn \
--env_name=Overcooked \
--verbose=False \
--is_multi_agent=True \
--log_dir=~/logs/minimax \
--log_interval=10 \
--from_last_checkpoint=False \
--checkpoint_interval=1000 \
--archive_interval=0 \
--archive_init_checkpoint=False \
--test_interval=100 \
--n_students=2 \
--n_parallel=32 \
--n_eval=1 \
--n_rollout_steps=400 \
--lr=0.0003 \
--lr_anneal_steps=0 \
--max_grad_norm=0.5 \
--adam_eps=1e-05 \
--track_env_metrics=True \
--discount=0.999 \
--n_unroll_rollout=10 \
--render=False \
--ued_score=relative_regret \
--student_gae_lambda=0.98 \
--teacher_discount=0.999 \
--teacher_lr_anneal_steps=0 \
--teacher_gae_lambda=0.98 \
--student_entropy_coef=0.01 \
--student_value_loss_coef=0.5 \
--student_n_unroll_update=5 \
--student_ppo_n_epochs=8 \
--student_ppo_n_minibatches=4 \
--student_ppo_clip_eps=0.2 \
--student_ppo_clip_value_loss=True \
--teacher_entropy_coef=0.01 \
--teacher_value_loss_coef=0.5 \
--teacher_n_unroll_update=5 \
--teacher_ppo_n_epochs=8 \
--teacher_ppo_n_minibatches=4 \
--teacher_ppo_clip_eps=0.2 \
--teacher_ppo_clip_value_loss=True \
--student_recurrent_arch=s5 \
--student_recurrent_hidden_dim=64 \
--student_hidden_dim=64 \
--student_n_hidden_layers=3 \
--student_n_conv_layers=3 \
--student_n_conv_filters=32 \
--student_n_scalar_embeddings=4 \
--student_scalar_embed_dim=5 \
--student_s5_n_blocks=2 \
--student_s5_n_layers=2 \
--student_s5_layernorm_pos=pre \
--student_s5_activation=half_glu1 \
--student_agent_kind=mappo \
--teacher_model_name=default_teacher_cnn \
--teacher_recurrent_arch=lstm \
--teacher_recurrent_hidden_dim=64 \
--teacher_hidden_dim=64 \
--teacher_n_hidden_layers=1 \
--teacher_n_conv_filters=128 \
--teacher_scalar_embed_dim=10 \
--overcooked_height=6 \
--overcooked_width=9 \
--overcooked_n_walls=5 \
--overcooked_normalize_obs=True \
--overcooked_max_steps=400 \
--overcooked_random_reset=False \
--overcooked_ued_replace_wall_pos=True \
--overcooked_ued_fixed_n_wall_steps=False \
--overcooked_ued_first_wall_pos_sets_budget=True \
--overcooked_ued_noise_dim=50 \
--overcooked_ued_n_walls=15 \
--overcooked_ued_normalize_obs=True \
--n_shaped_reward_updates=30000 \
--test_n_episodes=10 \
--test_env_names=Overcooked-CoordRing6_9,Overcooked-ForcedCoord6_9,Overcooked-CounterCircuit6_9,Overcooked-AsymmAdvantages6_9,Overcooked-CrampedRoom6_9 \
--overcooked_test_normalize_obs=True \
--xpid=SEED_${seed}_paired-overcooked6x9w5_ld50_rb-r2s_32p_1e_400t_ae1e-05_sr-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc3se5ba_re_lpr_ahg1_s5_h64nb2nl2_tch_ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98pc0.2_h64cf128fc1se10ba_re_lstm_h64_0