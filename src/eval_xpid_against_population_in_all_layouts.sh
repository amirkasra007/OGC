DEFAULTVALUE=4
device="${1:-$DEFAULTVALUE}"
NAME=$2
XPID=$3

for env in "Overcooked-CoordRing6_9" "Overcooked-ForcedCoord6_9" "Overcooked-CounterCircuit6_9" "Overcooked-AsymmAdvantages6_9" "Overcooked-CrampedRoom6_9";
do
    echo "Evaluating ${NAME} against population in ${env} for xpid ${XPID}"
    CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python3 -m minimax.evaluate_against_population \
    --xpid=${XPID} \
    --env_names=${env} \
    --population_json="populations/fcp/${env}/population.json" \
    --n_episodes=100
done
