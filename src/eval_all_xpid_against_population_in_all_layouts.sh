DEFAULTVALUE=1
device="${1:-$DEFAULTVALUE}"

# "Overcooked-CoordRing6_9" "Overcooked-ForcedCoord6_9" "Overcooked-CounterCircuit6_9" "Overcooked-AsymmAdvantages6_9" "Overcooked-CrampedRoom6_9"
# ./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CoordRing6_9 seed1_eval SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
# ./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-ForcedCoord6_9 seed1_eval SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
# ./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CounterCircuit6_9 seed1_eval SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
# ./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-AsymmAdvantages6_9 seed1_eval SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
# ./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CrampedRoom6_9 seed1_eval SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0


./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CoordRing6_9 SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-ForcedCoord6_9 SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CounterCircuit6_9 SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-AsymmAdvantages6_9 SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CrampedRoom6_9 SEED_1_dr-overcooked6x9w15_fs_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr0.0003g0.999cv0.5ce0.01e8mb4l0.98_pc0.2_h64cf32fc2se5ba_re_lstm_h64__SoftMoE_4E_32S___0

# #!/bin/bash

# DEFAULT_DEVICE=4
# device="${1:-$DEFAULT_DEVICE}"   # GPU index (default = 4)
# name="$2"                        # Used to label this evaluation run (e.g., seed1_eval)
# xpid="$3"                        # XPID of the agent to evaluate (e.g., SEED_1_...)

# # Layouts to evaluate against population
# layouts=(
#   "Overcooked-CoordRing6_9"
#   "Overcooked-ForcedCoord6_9"
#   "Overcooked-CounterCircuit6_9"
#   "Overcooked-AsymmAdvantages6_9"
#   "Overcooked-CrampedRoom6_9"
# )

# for env in "${layouts[@]}"; do
#   echo "Evaluating ${name} against population in ${env} for xpid ${xpid}"
#   CUDA_VISIBLE_DEVICES=${device} \
#   LD_LIBRARY_PATH="" \
#   nice -n 5 \
#   python3 -m minimax.evaluate_against_population \
#     --xpid="${xpid}" \
#     --xpid_prefix="${name}" \
#     --env_names="${env}" \
#     --population_json="populations/fcp/${env}/population.json" \
#     --n_episodes=100
# done
