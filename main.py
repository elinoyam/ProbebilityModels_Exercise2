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

    lidstone_model_010 = ProbabilityModel(training_set.words,model_type="Lidstone", gamma=0.10)
    lidstone_propability_input = lidstone_model_010.score(files_handler.input_word)
    files_handler.write_to_output_file(14, lidstone_propability_input)
    
    lidstone_propability_unseen = lidstone_model_010.score('unseen-word')
    files_handler.write_to_output_file(15, lidstone_propability_unseen)
    
    lidstone_model_001 = ProbabilityModel(training_set.words,model_type="Lidstone",model_type="Lidstone", gamma=0.01)
    lidstone_preplexity_001 = lidstone_model_010.perplexity(validation_set)
    files_handler.write_to_output_file(16, lidstone_preplexity_001)
    
    lidstone_preplexity_010 = lidstone_model_010.perplexity(validation_set)
    files_handler.write_to_output_file(17, lidstone_preplexity_010)
    
    lidstone_model_100 = ProbabilityModel(training_set.words,model_type="Lidstone", gamma=1.00)
    lidstone_preplexity_100 = lidstone_model_010.perplexity(validation_set)
    files_handler.write_to_output_file(18, lidstone_preplexity_100)

    min_preplexity, best_gamma = math.inf, -1
    for gamma in range(0, 2, 0.01):
        lidstone_model = ProbabilityModel(training_set.words,model_type="Lidstone", gamma=gamma)
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

    heldout_model = ProbabilityModel(small_training_set.words, model_type="HeldOut")
    heldout_model.set_held_out_data(small_training_set, held_out_set)

    input_word_prob = heldout_model.score(files_handler.input_word)
    files_handler.write_to_output_file(23, input_word_prob)
    unseen_word_prob = heldout_model.score('unseen-word')
    files_handler.write_to_output_file(24, unseen_word_prob)


    test_set, _ = files_handler.get_vocabulary_from_file('test', split_size=1.0)
    test_set_size = test_set.total()
    files_handler.write_to_output_file(25, test_set_size)

    lidstone_best_model = ProbabilityModel(training_set.words,model_type="Lidstone", gamma=best_gamma)
    lidstone_best_preplexity = lidstone_best_model.perplexity(test_set)
    files_handler.write_to_output_file(26, lidstone_best_preplexity)

    held_out_perplexity = heldout_model.perplexity(test_set)
    files_handler.write_to_output_file(27, held_out_perplexity)

    better_model = 'L' if lidstone_best_preplexity > held_out_perplexity else 'H'
    files_handler.write_to_output_file(28, better_model)










