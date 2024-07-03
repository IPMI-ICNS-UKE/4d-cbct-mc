from collections import Sequence


def convert_dict_values(d: dict, types, converter):
    if isinstance(d, (Sequence, set)):
        seq_type: type = type(d)
        return seq_type(
            [convert_dict_values(_d, types=types, converter=converter) for _d in d]
        )

    elif isinstance(d, types):
        return converter(d)

    elif isinstance(d, dict):
        # copy dict so we do not modify the original dict
        d = d.copy()
        for key, _d in d.items():
            d[key] = convert_dict_values(_d, types=types, converter=converter)

        return d

    else:
        return d


if __name__ == "__main__":
    d = {"a": [1, 2], "b": 3, "c": {"d": 4, "f": 5}, "g": {6, 7, 8}}
    dd = convert_dict_values(d, int, lambda x: f"{x:.6f}")
