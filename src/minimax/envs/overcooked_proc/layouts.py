# Edited from JaxMarl: https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked


import json
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

cramped_room = {
    "height": 4,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4,
                           5, 9,
                           10, 14,
                           15, 16, 17, 18, 19]),
    "agent_idx": jnp.array([6, 8]),
    "goal_idx": jnp.array([18]),
    "plate_pile_idx": jnp.array([16]),
    "onion_pile_idx": jnp.array([5, 9]),
    "pot_idx": jnp.array([2])
}
asymm_advantages = {
    "height": 5,
    "width": 9,
    "wall_idx": jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                           9, 11, 12, 13, 14, 15, 17,
                           18, 22, 26,
                           27, 31, 35,
                           36, 37, 38, 39, 40, 41, 42, 43, 44]),
    "agent_idx": jnp.array([29, 32]),
    "goal_idx": jnp.array([12, 17]),
    "plate_pile_idx": jnp.array([39, 41]),
    "onion_pile_idx": jnp.array([9, 14]),
    "pot_idx": jnp.array([22, 31])
}
coord_ring = {
    "height": 5,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4,
                           5, 9,
                           10, 12, 14,
                           15, 19,
                           20, 21, 22, 23, 24]),
    "agent_idx": jnp.array([7, 11]),
    "goal_idx": jnp.array([22]),
    "plate_pile_idx": jnp.array([10]),
    "onion_pile_idx": jnp.array([15, 21]),
    "pot_idx": jnp.array([3, 9])
}
forced_coord = {
    "height": 5,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4,
                           5, 7, 9,
                           10, 12, 14,
                           15, 17, 19,
                           20, 21, 22, 23, 24]),
    "agent_idx": jnp.array([11, 8]),
    "goal_idx": jnp.array([23]),
    "onion_pile_idx": jnp.array([5, 10]),
    "plate_pile_idx": jnp.array([15]),
    "pot_idx": jnp.array([3, 9])
}

# Example of layout provided as a grid
counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

asymm_advantages_6_9 = """
WWWWWWWWW
O WXWOW X
W   P A W
WA  P   W
WWWBWBWWW
WWWWWWWWW
"""

counter_circuit_6_9 = """
WWWPPWWWW
W A    WW
B WWWW XW
W     AWW
WWWOOWWWW
WWWWWWWWW
"""

forced_coord_6_9 = """
WWWPWWWWW
OAWAPWWWW
O W WWWWW
B W WWWWW
WWWXWWWWW
WWWWWWWWW
"""

cramped_room_6_9 = """
WWPWWWWWW
OAA OWWWW
W   WWWWW
WBWXWWWWW
WWWWWWWWW
WWWWWWWWW
"""

coord_ring_6_9 = """
WWWPWWWWW
WA APWWWW
B W WWWWW
O   WWWWW
WOXWWWWWW
WWWWWWWWW
"""

quad_6_9 = """
WWWWWWWWW
WWPA  WWW
WWB  AWWW
WWWOXOWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_1 = """
WWWPWWWWW
WWBA  WWW
WWO  AWWW
WWWXOWWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_2 = """
WWWBPWWWW
WWOA  WWW
WWX  AWWW
WWWOWWWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_3 = """
WWWOBPWWW
WWXA  WWW
WWO  AWWW
WWWWWWWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_4 = """
WWWXOBWWW
WWOA  PWW
WWW  AWWW
WWWWWWWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_5 = """
WWWOXOWWW
WWWA  BWW
WWW  APWW
WWWWWWWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_6 = """
WWWWOXWWW
WWWA  OWW
WWW  ABWW
WWWWWPWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_7 = """
WWWWWOWWW
WWWA  XWW
WWW  AOWW
WWWWPBWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_8 = """
WWWWWWWWW
WWWA  OWW
WWW  AXWW
WWWPBOWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9_9 = """
WWWWWWWWW
WWWA  WWW
WWP  AOWW
WWWBOXWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9T = """
WWWOXOWWW
WWW  APWW
WWWA  BWW
WWWWWWWWW
WWWWWWWWW
WWWWWWWWW
"""

quad_6_9M = """
WWWOXOWWW
WWB  APWW
WWPA  BWW
WWWOXOWWW
WWWWWWWWW
WWWWWWWWW
"""

asymm_advantages_10_15 = """
WWWWWWWWWWWWWWW
O WXWOW XWWWWWW
W   P A WWWWWWW
WA  P   WWWWWWW
WWWBWBWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
"""

asymm_advantages_B_10_15 = """
WWWWWWWWWWWWWWW
O WXWOW XWWWWWW
W   P A WWWWWWW
W   W   WWWWWWW
WA  P   WWWWWWW
WWWBWBWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
"""

asymm_advantages_M_10_15 = """
WWWWWWWWWWWWWWW
O  WXWOW  XWWWW
W    P A  WWWWW
WA   W    WWWWW
W    P    WWWWW
WWWBWWWBWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
"""

asymm_advantages_L_10_15 = """
WWWWWWWWWWWWWWW
O   WXWOW   XWW
W     P A   WWW
WA    W     WWW
W     W     WWW
W     P     WWW
WWWBWWWWWBWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
WWWWWWWWWWWWWWW
"""

asymm_advantages_m = """
WWWWWWWWW
O WOWXW X
W   P A W
WA  P   W
WWWBWBWWW
"""

asymm_advantages_m1 = """
WWWWWWWWW
X WXWOW O
W   P A W
WA  P   W
WWWBWBWWW
"""

asymm_advantages_m2 = """
WWWWWWWWW
O WXWOW X
WA  P A W
W   P   W
WWWBWBWWW
"""

asymm_advantages_m3 = """
WWWWWWWWW
O WXWOW X
W   P  AW
WA  P   W
WWWBWBWWW
"""

asymm_advantages_m4 = """
WWWWWWWWW
O WXWOW X
W   P A W
WA  B   W
WWWPWBWWW
"""

asymm_advantages_m5 = """
WWWWWWWWW
O WXWOW X
W   B A W
WA  P   W
WWWPWBWWW
"""

asymm_advantages_m6 = """
WWWWWWWWW
O WXWOW X
W   P A W
WA  P   W
WWBWWBWWW
"""

asymm_advantages_m7 = """
WWWWWWWWW
O WXWOW X
W   P A W
WA  P   W
WWBWWWBWW
"""

asymm_advantages_m8 = """
WWWWWWWWW
O WXWOW X
W   P A W
WA  P   W
WBWWWWBWW
"""


def layout_grid_to_onehot_dict(grid):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    O: onion pile
    P: pot location
    ' ' (space) : empty cell
    """

    rows = grid.split('\n')

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx", "goal_idx",
            "plate_pile_idx", "onion_pile_idx", "pot_idx", "empty_table_idx"]
    symbol_to_key = {"W": "wall_idx",
                     "A": "agent_idx",
                     "X": "goal_idx",
                     "B": "plate_pile_idx",
                     "O": "onion_pile_idx",
                     "P": "pot_idx"}

    layout_dict = {key: [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])

    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            # if obj in symbol_to_key.keys():
            #     # Add object
            #     layout_dict[symbol_to_key[obj]].append(idx)

            if obj == "A":
                # Agent
                layout_dict["agent_idx"].append(idx)

            if obj == "X":
                # Goal
                layout_dict["goal_idx"].append(1)
            else:
                layout_dict["goal_idx"].append(0)

            if obj == "B":
                # Plate pile
                layout_dict["plate_pile_idx"].append(1)
            else:
                layout_dict["plate_pile_idx"].append(0)

            if obj == "O":
                # Onion pile
                layout_dict["onion_pile_idx"].append(1)
            else:
                layout_dict["onion_pile_idx"].append(0)

            if obj == "P":
                # Pot location
                layout_dict["pot_idx"].append(1)
            else:
                layout_dict["pot_idx"].append(0)

            if obj in ["X", "B", "O", "P", "W"]:
                # These objects are also walls technically
                layout_dict["wall_idx"].append(1)
            else:
                layout_dict["wall_idx"].append(0)

            if obj == "W":
                # Goal
                layout_dict["empty_table_idx"].append(1)
            else:
                layout_dict["empty_table_idx"].append(0)
            # elif obj == " ":
            #     # Empty cell
            #     continue

    for key in layout_dict.keys():
        # Transform lists to arrays
        layout_dict[key] = jnp.array(layout_dict[key], dtype=jnp.uint8)

    return FrozenDict(layout_dict)


def layout_grid_to_dict(grid):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    O: onion pile
    P: pot location
    ' ' (space) : empty cell
    """

    rows = grid.split('\n')

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx", "goal_idx",
            "plate_pile_idx", "onion_pile_idx", "pot_idx"]
    symbol_to_key = {"W": "wall_idx",
                     "A": "agent_idx",
                     "X": "goal_idx",
                     "B": "plate_pile_idx",
                     "O": "onion_pile_idx",
                     "P": "pot_idx"}

    layout_dict = {key: [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])

    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            if obj in symbol_to_key.keys():
                # Add object
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["X", "B", "O", "P"]:
                # These objects are also walls technically
                layout_dict["wall_idx"].append(idx)
            elif obj == " ":
                # Empty cell
                continue

    for key in symbol_to_key.values():
        # Transform lists to arrays
        layout_dict[key] = jnp.array(layout_dict[key])

    return FrozenDict(layout_dict)


# load all_lvl_strs.json
# all_lvls_strs = json.load(
#     open("jaxmarl/environments/overcooked/10x15_all_lvl_strs.json", "r"))
# gan_mlp_layouts = all_lvls_strs["gan_milp"]

# automatic_overcooked_layouts_10_15 = {
#     str(k): layout_grid_to_dict(v) for k, v in enumerate(gan_mlp_layouts)
# }

# all_lvls_strs = json.load(
#     open("jaxmarl/environments/overcooked/6x9_all_lvl_strs.json", "r"))
# gan_mlp_layouts = all_lvls_strs["gan_milp"]

# automatic_overcooked_layouts_6_9 = {
#     str(k): layout_grid_to_dict(v) for k, v in enumerate(gan_mlp_layouts)
# }

# all_lvls_strs = json.load(
#     open("jaxmarl/environments/overcooked/6x9_all_lvl_strs_simple.json", "r"))

# gan_mlp_layouts = all_lvls_strs["gan_milp"]

# automatic_overcooked_layouts_6_9_simple = {
#     str(k): layout_grid_to_dict(v) for k, v in enumerate(gan_mlp_layouts)
# }


overcooked_layouts = {
    "cramped_room": FrozenDict(cramped_room),
    "asymm_advantages": FrozenDict(asymm_advantages),
    "asymm_advantages_m": FrozenDict(asymm_advantages),
    "coord_ring": FrozenDict(coord_ring),
    "forced_coord": FrozenDict(forced_coord),
    "counter_circuit": layout_grid_to_dict(counter_circuit_grid),
    "asymm_advantages_6_9": layout_grid_to_dict(asymm_advantages_6_9),
    "counter_circuit_6_9": layout_grid_to_dict(counter_circuit_6_9),
    "forced_coord_6_9": layout_grid_to_dict(forced_coord_6_9),
    "cramped_room_6_9": layout_grid_to_dict(cramped_room_6_9),
    "coord_ring_6_9": layout_grid_to_dict(coord_ring_6_9),
    "quad_6_9": layout_grid_to_dict(quad_6_9),
    "quad_6_9_1": layout_grid_to_dict(quad_6_9_1),
    "quad_6_9_2": layout_grid_to_dict(quad_6_9_2),
    "quad_6_9_3": layout_grid_to_dict(quad_6_9_3),
    "quad_6_9_4": layout_grid_to_dict(quad_6_9_4),
    "quad_6_9_5": layout_grid_to_dict(quad_6_9_5),
    "quad_6_9_6": layout_grid_to_dict(quad_6_9_6),
    "quad_6_9_7": layout_grid_to_dict(quad_6_9_7),
    "quad_6_9_8": layout_grid_to_dict(quad_6_9_8),
    "quad_6_9_9": layout_grid_to_dict(quad_6_9_9),
    "quad_6_9T": layout_grid_to_dict(quad_6_9T),
    "quad_6_9M": layout_grid_to_dict(quad_6_9M),
    "asymm_advantages_m": layout_grid_to_dict(asymm_advantages_m),
    "asymm_advantages_m1": layout_grid_to_dict(asymm_advantages_m1),
    "asymm_advantages_m2": layout_grid_to_dict(asymm_advantages_m2),
    "asymm_advantages_m3": layout_grid_to_dict(asymm_advantages_m3),
    "asymm_advantages_m4": layout_grid_to_dict(asymm_advantages_m4),
    "asymm_advantages_m5": layout_grid_to_dict(asymm_advantages_m5),
    "asymm_advantages_m6": layout_grid_to_dict(asymm_advantages_m6),
    "asymm_advantages_m7": layout_grid_to_dict(asymm_advantages_m7),
    "asymm_advantages_m8": layout_grid_to_dict(asymm_advantages_m8),
    "asymm_advantages_10_15": layout_grid_to_dict(asymm_advantages_10_15),
    "asymm_advantages_B_10_15": layout_grid_to_dict(asymm_advantages_B_10_15),
    "asymm_advantages_M_10_15": layout_grid_to_dict(asymm_advantages_M_10_15),
    "asymm_advantages_L_10_15": layout_grid_to_dict(asymm_advantages_L_10_15),
}
