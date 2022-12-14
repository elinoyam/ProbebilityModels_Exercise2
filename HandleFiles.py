from ProbabilityModel import prob_model
from Vocabulary import VocabularySet
from main import vocabulary_size, uniform_probability

class FilesHandler:

    def __init__(self, args):
        self.development_file = args[1]
        self.test_file = args[2]
        self.input_word = args[3]
        self.output_file = args[4]


    def write_to_output_file(self, row_number, value):
        with open(file=self.output_file, mode='a') as output_file:
            output_file.write(f'Output{row_number} {value}\n')

    def initialize_output_file(self):
        with open(file=self.output_file, mode='a') as output_file:
            output_file.write(f'Eden Mironi Id-____, Elinoy Amar Id-318532132\n')
            output_file.write(f'Output1 {self.development_file}\n')
            output_file.write(f'Output2 {self.test_file}\n')
            output_file.write(f'Output3 {self.input_word}\n')
            output_file.write(f'Output4 {self.output_file}\n')
            output_file.write(f'Output5 {vocabulary_size}\n')
            output_file.write(f'Output5 {uniform_probability}\n')

    def get_vocabulary_from_file(self, input_file = 'development', split_size=0.9):
        file_name = self.development_file if input_file == 'development' else self.test_file
        training_set, validation_set = VocabularySet(), VocabularySet()
        with open(file_name, mode='r') as file:
            list_of_words = file.read().split()
            training_set_size = round(len(list_of_words) * split_size)
            word_index = 0
            for word in list_of_words:
                if word_index < training_set_size:
                    training_set[word] += 1
                else:
                    validation_set[word] += 1

                word_index += 1

        return training_set, validation_set


