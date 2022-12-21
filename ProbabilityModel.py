import main
import math

class ProbabilityModel:

    def __init__(self, counter, model_type="Lidstone", gamma=0):
        self.Counter = counter
        self.model_type = model_type
        self.gamma = gamma
        self.r_classes = {}
        self.T_r = {}
        self.held_out_probability = {}
    
    def lidstone_propabilty(self, word):
        word_count = self.Counter[word]
        return (word_count + self.gamma) / (self.Counter.total() + main.vocabulary_size  * self.gamma)

    def heldout_propabilty(self, word):
        return self.held_out_probability[self.Counter[word]] if word in self.Counter else self.held_out_probability[0]

    def score(self, word):
        return self.lidstone_propabilty(word) if self.model_type == "Lidstone" else self.heldout_propabilty(word)

    def perplexity(self, validation_vocab):
        ''' Compute model's perplexity over a given test vocabulary '''

        probabilities = [self.score(word) for word in validation_vocab.keys()]

        return pow(2, ((-1 / validation_vocab.total()) * sum(map(math.log, probabilities))))
    
    def set_held_out_data(self, small_training_set, held_out_set):
        # compute each seen word in T probability
        # group words by the number of appearances in small_training_set
        for key, r in small_training_set.get_words_by_appearances():
            if r not in self.r_classes:     # first word with r appearances
                self.r_classes[r] = [key]
            else:   # add the word to the existing words list with the same appearances count
                self.r_classes[r].append(key)

        # get all the words not in small_training_set and yes in held_out_set
        for key, r in held_out_set.get_words_by_appearances():
            if key not in small_training_set:
                if 0 not in self.r_classes:
                    self.r_classes[0] = [key]
                else:
                    self.r_classes[0].append(key)

        # for each group of appearances - count the number of appearances in held_out_set
        for r, words_list in self.r_classes:
            for word in words_list:
                if r not in self.T_r:  # first word with r appearances
                    self.T_r[r] = held_out_set[word]
                else:
                    self.T_r[r] += held_out_set[word]

        # calculate each r_class probability (all the words in each r class will be with the same probability)
        for r, total in self.T_r:
            # if r not in held_out_probability:  # first word with r appearances
            self.held_out_probability[r] = total / (len(self.r_classes[r]) * held_out_set.total())

prob_model = ProbabilityModel()


