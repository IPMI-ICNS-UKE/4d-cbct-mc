import re


def natural_sort_key(s, pattern=re.compile("([0-9]+)")):
    """A key for Python's built-in sorted(..., key=natural_sort_key). Useful
    for sorting, e.g., filename sequences without leading zeros like.

    ['img1.png', 'img2.png', 'img10.png'].
    :param s: Current element of sequence
    :param pattern: A compiled regex pattern to find the numbers
    :return: Sorting key for current element
    """
    if not isinstance(s, str):
        s = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in pattern.split(s)]
