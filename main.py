import math
from sys import argv
from HandleFiles import FilesHandler
from ProbabilityModel import ProbabilityModel

vocabulary_size = 300000
uniform_probability = 1/vocabulary_size

if __name__ == '__main__':
    # prob_model = ProbabilityModel()
    files_handler = FilesHandler(argv)

    files_handler.initialize_output_file()  # start filling the output file, will print rows 1-6
    training_set, validation_set = files_handler.get_vocabulary_from_file('development')
    validation_set_size = validation_set.total()
    training_set_size = training_set.total()
    dev_vocab_size = training_set_size + validation_set_size
    files_handler.write_to_output_file(7, dev_vocab_size)   # total number of events in the development set |S|
    files_handler.write_to_output_file(8, validation_set_size)
    files_handler.write_to_output_file(9, training_set_size)
    files_handler.write_to_output_file(10, len(training_set))
    files_handler.write_to_output_file(11, training_set[files_handler.input_word])

    input_word_freq = training_set[files_handler.input_word] / training_set.total()
    files_handler.write_to_output_file(12, input_word_freq)
    


    # line 20
    min_preplexity, best_gamma = math.inf, -1
    for gamma in range(0, 2, 0.01):
        lidstone_model = ProbabilityModel(training_set.words, gamma)
        preplexity = lidstone_model.perplexity(validation_set)
        if preplexity < min_preplexity:
            min_preplexity = preplexity
            best_gamma = gamma
    files_handler.write_to_output_file(20, min_preplexity)








