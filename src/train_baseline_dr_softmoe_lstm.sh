#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ogc

export LD_LIBRARY_PATH=$HOME/cudnn-8.9.5/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=$HOME/cudnn-8.9.5/include
export CUDA_HOME=/usr/local/cuda-12.2
export LD_PRELOAD=$HOME/cudnn-8.9.5/lib/libcudnn.so

DEFAULTSEED=3
seed="${1:-$DEFAULTSEED}"

# Debug output:
echo "Python: $(which python)"
python --version
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.40 nice -n 5 python3 -m minimax.train \
--wandb_mode=online \
--wandb_project=overcooked-minimax-jax \
--wandb_entity=${WANDB_ENTITY} \
--seed=${seed} \
--agent_rl_algo=ppo \
--n_total_updates=30000 \
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
--checkpoint_interval=1000 \
--archive_interval=0 \
--archive_init_checkpoint=False \
--test_interval=100 \
--n_students=1 \
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
--student_gae_lambda=0.98 \
--student_entropy_coef=0.01 \
--student_value_loss_coef=0.5 \
--student_n_unroll_update=5 \
--student_ppo_n_epochs=8 \
--student_ppo_n_minibatches=4 \
--student_ppo_clip_eps=0.2 \
--student_ppo_clip_value_loss=True \
--student_recurrent_arch=lstm \
--student_recurrent_hidden_dim=64 \
--student_hidden_dim=64 \
--student_n_hidden_layers=2 \
--student_is_soft_moe=True \
--student_soft_moe_num_experts=4 \
--student_soft_moe_num_slots=32 \
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
--n_shaped_reward_updates=30000 \
--test_n_episodes=10 \
--test_env_names=Overcooked-CoordRing6_9,Overcooked-ForcedCoord6_9,Overcooked-CounterCircuit6_9,Overcooked-AsymmAdvantages6_9,Overcooked-CrampedRoom6_9 \
--overcooked_test_normalize_obs=True \
--xpid=SEED_${seed}_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
