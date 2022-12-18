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

    # training_set['unseen-word'] will return 0 if not exists, else calculate as the usual formula
    unseen_word_freq = training_set['unseen-word'] / training_set.total() 
    files_handler.write_to_output_file(13, input_word_freq)

    lidstone_model_010 = ProbabilityModel(training_set.words, 0.10)
    lidstone_propability_input = lidstone_model_010.score(files_handler.input_word)
    files_handler.write_to_output_file(14, lidstone_propability_input)
    
    lidstone_propability_unseen = lidstone_model_010.score('unseen-word')
    files_handler.write_to_output_file(15, lidstone_propability_unseen)
    
    lidstone_model_001 = ProbabilityModel(training_set.words, 0.01)
    lidstone_preplexity_001 = lidstone_model_010.perplexity(validation_set)
    files_handler.write_to_output_file(16, lidstone_preplexity_001)
    
    lidstone_preplexity_010 = lidstone_model_010.perplexity(validation_set)
    files_handler.write_to_output_file(17, lidstone_preplexity_010)
    
    lidstone_model_100 = ProbabilityModel(training_set.words, 1.00)
    lidstone_preplexity_100 = lidstone_model_010.perplexity(validation_set)
    files_handler.write_to_output_file(18, lidstone_preplexity_100)

    min_preplexity, best_gamma = math.inf, -1
    for gamma in range(0, 2, 0.01):
        lidstone_model = ProbabilityModel(training_set.words, gamma)
        preplexity = lidstone_model.perplexity(validation_set)
        if preplexity < min_preplexity:
            min_preplexity = preplexity
            best_gamma = gamma
            
    files_handler.write_to_output_file(19, best_gamma)
    files_handler.write_to_output_file(20, min_preplexity)

    # held_out_model
    small_training_set, held_out_set = files_handler.get_vocabulary_from_file('development', split_size=0.5)
    held_out_set_size = held_out_set.total()
    small_training_set_size = small_training_set.total()
    files_handler.write_to_output_file(21, small_training_set_size)
    files_handler.write_to_output_file(22, held_out_set_size)

    # compute each seen word in T probability
    r_classes = {}

    # group words by the number of appearances in small_training_set
    for key, r in small_training_set.get_words_by_appearances():
        if r not in r_classes:     # first word with r appearances
            r_classes[r] = [key]
        else:   # add the word to the existing words list with the same appearances count
            r_classes[r].append(key)

    # get all the words not in small_training_set and yes in held_out_set
    for key, r in held_out_set.get_words_by_appearances():
        if key not in small_training_set:
            if 0 not in r_classes:
                r_classes[0] = [key]
            else:
                r_classes[0].append(key)

    T_r = {}

    # for each group of appearances - count the number of appearances in held_out_set
    for r, words_list in r_classes:
        for word in words_list:
            if r not in T_r:  # first word with r appearances
                T_r[r] = held_out_set[word]
            else:
                T_r[r] += held_out_set[word]

    # calculate each r_class probability (all the words in each r class will be with the same probability)
    held_out_probability = {}
    for r, total in T_r:
        # if r not in held_out_probability:  # first word with r appearances
        held_out_probability[r] = total / (len(r_classes[r]) * held_out_set_size)

    input_word_prob = held_out_probability[small_training_set[files_handler.input_word]] if files_handler.input_word in small_training_set else held_out_probability[0]
    files_handler.write_to_output_file(23, input_word_prob)
    unseen_word_prob = held_out_probability[0]
    files_handler.write_to_output_file(24, unseen_word_prob)









