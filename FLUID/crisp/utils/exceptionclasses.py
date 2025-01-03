class TargetHasZeroVarianceError(Exception):
    """Exception raised if target variable variance is 0.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Chosen target variable has 0 variance in training data, cannot do inference"):
        self.message = message
        super().__init__(self.message)
