DEFAULTVALUE=4
DEFAULTSEED=1
device="${1:-$DEFAULTVALUE}"
seed="${2:-$DEFAULTSEED}"
echo "Using device ${device} and seed ${seed}"

./train_baseline_p_plr_lstm.sh $device $seed
./train_baseline_p_accel_lstm.sh $device $seed
./train_baseline_pop_paired_lstm.sh $device $seed
./train_baseline_dr_lstm.sh $device $seed
