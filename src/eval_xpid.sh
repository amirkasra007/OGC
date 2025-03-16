DEFAULTVALUE=4
device="${1:-$DEFAULTVALUE}"
CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python3 -m minimax.evaluate \
--xpid=$2 \
--env_names=Overcooked-CoordRing6_9,Overcooked-ForcedCoord6_9,Overcooked-CounterCircuit6_9,Overcooked-AsymmAdvantages6_9,Overcooked-CrampedRoom6_9 \
--n_episodes=1000