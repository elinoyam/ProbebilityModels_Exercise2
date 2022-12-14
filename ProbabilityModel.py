import main
import math

class ProbabilityModel:

    def __init__(self, counter, gamma):
        self.gamma = gamma
        self.Counter = counter

    def score(self, word):

        word_count = self.Counter[word]
        return (word_count + self.gamma) / (self.Counter.total() + main.vocabulary_size  * self.gamma)

    def perplexity(self, validation_vocab):
        ''' Compute model's perplexity over a given test vocabulary '''

        probabilities = [self.score(word) for word in validation_vocab.keys()]

        return pow(2, ((-1 / validation_vocab.total()) * sum(map(math.log, probabilities))))

prob_model = ProbabilityModel()


