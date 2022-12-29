import HandleFiles
import math

class ProbabilityModel:

    def __init__(self, counter, model_type="Lidstone", gamma=0.01, train_words = None, test_words = None ):
        self.Counter = counter
        self.model_type = model_type
        self.gamma = float(gamma)
        self.r_classes = {}
        self.T_r = {}
        self.held_out_probability = {}
        self.train_set = train_words
        self.test_set = test_words
        self.counter_total = self.Counter.total()
    
    def lidstone_propabilty(self, word): # len(trainingSet) , vocabularySize = 300000
        word_count = self.Counter[word] if word in self.Counter else 0
        return float(word_count + self.gamma) / (self.counter_total + HandleFiles.vocabulary_size * self.gamma)

    def heldout_propabilty(self, word):
        return self.held_out_probability[self.Counter[word]] if word in self.Counter else self.held_out_probability[0]

    def score(self, word):
        return self.lidstone_propabilty(word) if self.model_type == "Lidstone" else self.heldout_propabilty(word)

    def perplexity(self, validation_vocab):
        ''' Compute model's perplexity over a given test vocabulary '''

        prob_log_sum = 0
        for word in self.test_set:
            prob = self.score(word)
            if prob > 0:
                prob_log_sum += math.log(prob)

        return math.exp(-(prob_log_sum / len(self.test_set)))

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
        for r in self.r_classes.keys():
            words_list = self.r_classes[r]
            for word in words_list:
                if r not in self.T_r:  # first word with r appearances
                    self.T_r[r] = held_out_set[word]
                else:
                    self.T_r[r] += held_out_set[word]

        # calculate each r_class probability (all the words in each r class will be with the same probability)
        for r in self.T_r.keys():
            total = self.T_r[r]
            n_r = len(self.r_classes[r])
            if r == 0:
                n_r = HandleFiles.vocabulary_size - len(small_training_set.keys())
                
            # if r not in held_out_probability:  # first word with r appearances
            self.held_out_probability[r] = total / (n_r * held_out_set.total())



