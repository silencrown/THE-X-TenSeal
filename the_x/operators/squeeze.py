def unsqueeze(lst, dim):
    """
    Insert a new dimension at the specified position and convert the list to a multidimensional list.
    :param lst: The input list.
    :param dim: The dimension size of the newly added dimension.
    :return: A multidimensional list.
    """
    return [lst[i:i+dim] for i in range(0, len(lst), dim)]

def squeeze(lst):
    """
    Convert a multidimensional list to a list.
    :param lst: The input multidimensional list.
    :return: A list.
    """
    return [item for sublist in lst for item in sublist]