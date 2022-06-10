class CN2:

    def __init__(self, star_max_size=3, epsilon=0.5):
        self.data = None
        self.star_max_size = star_max_size
        self.epsilon = epsilon  # significance (between 0 and 1) @TODO error handling when user inputs sth stupid
        self._P = []  # subset on which we will train in every iteration (when it's 0 algorithm is done)
        self._selectors = []  # set of atomic selectros

    # @TODO - here will appear all the magic
    def fit(self, dataset):
        pass

    # @TODO
    def predict(self):
        pass

    def find_selectors(self):
        """
        This function finds all possible selectors in given dataset
        writes to self._selectors in this pattern:
        self._selectors = [('column1', 'value1'), ('column1', 'value2')]
        """
        possible_selectors = self.data.drop(columns='class')

        for column in possible_selectors:
            possible_values = set(self.data[column])
            for value in possible_values:
                self._selectors.append((column, value))
