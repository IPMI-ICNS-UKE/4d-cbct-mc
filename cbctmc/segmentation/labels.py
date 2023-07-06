LABELS = {
    0: "background",  # softmax group 1
    1: "upper_body_bones",  # softmax group 1
    2: "upper_body_muscles",  # softmax group 1
    3: "upper_body_fat",  # softmax group 1
    4: "liver",  # softmax group 1
    5: "stomach",  # softmax group 1
    6: "lung",  # softmax group 1
    7: "other",  # softmax group 1
    8: "lung_vessels",  # sigmoid
}

# LABELS = {
#     0: "background",  # softmax group 1
#     1: "upper_body_bones",  # softmax group 1
#     # 2: "upper_body_muscles",  # softmax group 1
#     # 3: "upper_body_fat",  # softmax group 1
#     # 4: "liver",  # softmax group 1
#     # 5: "stomach",  # softmax group 1
#     2: "lung",  # softmax group 1
#     3: "other",  # softmax group 1
#     4: "lung_vessels",  # sigmoid
# }

# LABELS = {
#     0: "background",  # softmax group 1
#     1: "upper_body_bones",  # softmax group 1
#     2: "upper_body_muscles",  # softmax group 1
#     3: "upper_body_fat",  # softmax group 1
#     4: "liver",  # softmax group 1
#     5: "stomach",  # softmax group 1
#     6: "lung",  # softmax group 1
#     7: "other",  # softmax group 1
# }


LABELS_TO_LOAD = [
    "upper_body_bones",
    "upper_body_muscles",
    "upper_body_fat",
    "liver",
    "stomach",
    "lung",
    "lung_vessels",
    "body",
]

N_LABELS = len(LABELS)


def get_label_index(label_name: str) -> int:
    return list(LABELS.values()).index(label_name)
