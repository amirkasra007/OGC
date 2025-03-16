DEFAULTVALUE=4
device="${1:-$DEFAULTVALUE}"

# "Overcooked-CoordRing6_9" "Overcooked-ForcedCoord6_9" "Overcooked-CounterCircuit6_9" "Overcooked-AsymmAdvantages6_9" "Overcooked-CrampedRoom6_9"
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CoordRing6_9 9SEED_9_dr-overcookedNonexNonewNone_fs_FIXcoord_ring_6_9_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr3e-5g0.99cv0.5ce0.01e5mb1l0.95_pc0.2_h64cf32fc2se5ba_re_0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-ForcedCoord6_9 9SEED_9_dr-overcookedNonexNonewNone_fs_FIXforced_coord_6_9_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr3e-5g0.99cv0.5ce0.01e5mb1l0.95_pc0.2_h64cf32fc2se5ba_re_0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CounterCircuit6_9 9SEED_9_dr-overcookedNonexNonewNone_fs_FIXcounter_circuit_6_9_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr3e-5g0.99cv0.5ce0.01e5mb1l0.95_pc0.2_h64cf32fc2se5ba_re_0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-AsymmAdvantages6_9 9SEED_9_dr-overcookedNonexNonewNone_fs_FIXasymm_advantages_6_9_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr3e-5g0.99cv0.5ce0.01e5mb1l0.95_pc0.2_h64cf32fc2se5ba_re_0
./eval_xpid_against_population_in_all_layouts.sh $device Overcooked-CrampedRoom6_9 9SEED_9_dr-overcookedNonexNonewNone_fs_FIXcramped_room_6_9_IMAGE-r1s_32p_1e_400t_ae1e-05-ppo_lr3e-5g0.99cv0.5ce0.01e5mb1l0.95_pc0.2_h64cf32fc2se5ba_re_0