def is_tuple(v, min_length=0):
    return isinstance(v, tuple) and len(v) >= min_length


def is_tuple_of_types(v, types, min_length=0):
    if not is_tuple(types):
        types = (types,)
    return is_tuple(v, min_length=max(1, min_length)) and all(
        isinstance(vv, t) for vv in v for t in types
    )


def is_tuple_of_tuples(v):
    return is_tuple_of_types(v, types=tuple)


def are_of_same_length(*args) -> bool:
    return len(set(map(len, args))) in (0, 1)
