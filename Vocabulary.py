from collections import Counter


class VocabularySet:

    """
        This class suppose to hold a vocabulary as a dictionary,
        with the option to set a minimum appearances criteria
    """

    def __init__(self, minimum_appearances=1):
        self.words = Counter()
        self.min_appearances = minimum_appearances

    def __len__(self):
        return len(self.words)

    def total(self):
        return self.words.total()

    def __contains__(self, item):
        return item in self.words

    def __getitem__(self, item):
        """
            Will return the number of appearance of a word if it's more than the minimum criteria,
            otherwise will return 0.
        """

        counts = self.words[item]
        return counts if counts > self.min_appearances else 0

    def insert(self, dict):
        """ This method will insert or update words counter in vocabulary. """
        self.words.update(dict)

    def get_words_by_appearances(self):
        return self.words.most_common()
