DEFAULTVALUE=4
device="${1:-$DEFAULTVALUE}"

seed=42

for layout in "coord_ring_6_9" "forced_coord_6_9" "cramped_room_6_9" "asymm_advantages_6_9" "counter_circuit_6_9";
do
    echo "layout is ${layout}:"
    CUDA_VISIBLE_DEVICES=${device} XLA_PYTHON_CLIENT_MEM_FRACTION=.40 LD_LIBRARY_PATH="" nice -n 5 python3 -m minimax.train \
    --wandb_mode=online \
    --wandb_project=overcooked-minimax-jax \
    --wandb_entity=${WANDB_ENTITY} \
    --seed=${seed} \
    --agent_rl_algo=ppo \
    --n_total_updates=1000 \
    --train_runner=dr \
    --n_devices=1 \
    --student_model_name=default_student_actor_cnn \
    --student_critic_model_name=default_student_critic_cnn \
    --env_name=Overcooked \
    --is_multi_agent=True \
    --verbose=False \
    --log_dir=~/logs/minimax \
    --log_interval=10 \
    --from_last_checkpoint=False \
    --checkpoint_interval=25 \
    --archive_interval=25 \
    --archive_init_checkpoint=False \
    --test_interval=50 \
    --n_students=1 \
    --n_parallel=100 \
    --n_eval=1 \
    --n_rollout_steps=400 \
    --lr=3e-4 \
    --lr_anneal_steps=0 \
    --max_grad_norm=0.5 \
    --adam_eps=1e-05 \
    --track_env_metrics=True \
    --discount=0.99 \
    --n_unroll_rollout=10 \
    --render=False \
    --student_gae_lambda=0.95 \
    --student_entropy_coef=0.01 \
    --student_value_loss_coef=0.5 \
    --student_n_unroll_update=5 \
    --student_ppo_n_epochs=5 \
    --student_ppo_n_minibatches=1 \
    --student_ppo_clip_eps=0.2 \
    --student_ppo_clip_value_loss=True \
    --student_hidden_dim=64 \
    --student_n_hidden_layers=3 \
    --student_n_conv_layers=3 \
    --student_n_conv_filters=32 \
    --student_n_scalar_embeddings=4 \
    --student_scalar_embed_dim=5 \
    --student_agent_kind=mappo \
    --overcooked_height=6 \
    --overcooked_width=9 \
    --overcooked_n_walls=15 \
    --overcooked_replace_wall_pos=True \
    --overcooked_sample_n_walls=True \
    --overcooked_normalize_obs=True \
    --overcooked_max_steps=400 \
    --overcooked_random_reset=False \
    --overcooked_fix_to_single_layout=${layout} \
    --n_shaped_reward_steps=3000000 \
    --test_n_episodes=10 \
    --test_env_names=Overcooked-CoordRing6_9,Overcooked-ForcedCoord6_9,Overcooked-CounterCircuit6_9,Overcooked-AsymmAdvantages6_9,Overcooked-CrampedRoom6_9 \
    --overcooked_test_normalize_obs=True \
    --xpid=9SEED_${seed}_dr-overcookedNonexNonewNone_fs_FIX${layout}_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr3e-5g0.99cv0.5ce0.01e5mb1l0.95_pc0.2_h64cf32fc2se5ba_re_0
done