DEFAULTVALUE=4
device="${1:-$DEFAULTVALUE}"

./baseline_train__8_seeds.sh $device coord_ring_6_9
./baseline_train__8_seeds.sh $device counter_circuit_6_9
./baseline_train__8_seeds.sh $device forced_coord_6_9
./baseline_train__8_seeds.sh $device cramped_room_6_9
./baseline_train__8_seeds.sh $device asymm_advantages_6_9