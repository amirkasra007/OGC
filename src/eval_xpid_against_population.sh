DEFAULTVALUE=4
# Overcooked-CoordRing6_9,Overcooked-ForcedCoord6_9,Overcooked-CounterCircuit6_9,Overcooked-AsymmAdvantages6_9,Overcooked-CrampedRoom6_9
device="${1:-$DEFAULTVALUE}"
ENV=Overcooked-AsymmAdvantages6_9 
XPID=$2
CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python3 -m minimax.evaluate_against_population \
--xpid=${XPID} \
--env_names=${ENV} \
--population_json="populations/fcp/${ENV}/population.json" \
--n_episodes=100
