class IndexNotBuiltError(ValueError, AttributeError):
    """Exception raised when attempting to find nearest neighbors
    before the ANN index is built.
    """

    pass


class NNDescentHyperplaneError(Exception):
    """Exception raised when NNDescent can't find a hyperplane.

    Usually occurs with small data.
    """

    pass
