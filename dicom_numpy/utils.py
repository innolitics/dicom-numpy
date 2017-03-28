def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    '''
    This function is implemented in Python 3.

    To support Python 2, we include our own implementation.
    '''
    return abs(a-b) <= max(rel_tol*max(abs(a), abs(b)), abs_tol)
