DEFAULTVALUE=4
ENV=Overcooked-CrampedRoom5_5 # Overcooked-CoordRing6_9,Overcooked-ForcedCoord6_9,Overcooked-CounterCircuit6_9,Overcooked-AsymmAdvantages6_9,Overcooked-CrampedRoom6_9
device="${1:-$DEFAULTVALUE}"

seed_max=8

for seed in `seq ${seed_max}`;
do
    CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python3 -m minimax.extract_fcp \
    --xpid=8SEED_${seed}_$2 \
    --env_names=${ENV} \
    --n_episodes=100 \
    --trained_seed=${seed}
done