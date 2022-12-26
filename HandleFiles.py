from Vocabulary import VocabularySet
from tabulate import tabulate

vocabulary_size = 300000
uniform_probability = 1/vocabulary_size

class FilesHandler:

    def __init__(self, args):
        self.development_file = args[1]
        self.test_file = args[2]
        self.input_word = args[3]
        self.output_file = args[4]


    def write_to_output_file(self, row_number, value):
        with open(file=self.output_file, mode='a') as output_file:
            output_file.write(f'Output{row_number} {value}\n')
    
    def write_table_to_output_file(self, row_number, values_list):
        with open(file=self.output_file, mode='a') as output_file:
            output_file.write(f'Output{row_number}\n')
            for row in values_list:
                print(row)
                printable_list = [f'{round(item, 5)}' for item in row]
                output_file.write("\t".join(printable_list) + "\n")
            # output_file.write(f'{tabulate(list)}\n')

    def initialize_output_file(self):
        with open(file=self.output_file, mode='w') as output_file:
            output_file.write(f'Eden Mironi Id-207856097, Elinoy Amar Id-318532132\n')
            output_file.write(f'Output1 {self.development_file}\n')
            output_file.write(f'Output2 {self.test_file}\n')
            output_file.write(f'Output3 {self.input_word}\n')
            output_file.write(f'Output4 {self.output_file}\n')
            output_file.write(f'Output5 {vocabulary_size}\n')
            output_file.write(f'Output5 {uniform_probability}\n')

    def get_vocabulary_from_file(self, input_file = 'development', split_size=0.9):
        file_name = self.development_file if input_file == 'development' else self.test_file
        training_set, validation_set = VocabularySet(), VocabularySet()
        list_of_all_words = []
        with open(file_name, mode='r') as file:
            for line in file:
                if line[:6] != '<TRAIN' and line[:5] != '<TEST':    # not a header line
                    list_of_all_words.extend(line.split())

            training_set_size = round(len(list_of_all_words) * split_size) if split_size < 1.0 else len(list_of_all_words)
            word_index = 0
            for word in list_of_all_words:
                if word_index < training_set_size:
                    training_set[word] += 1
                else:
                    validation_set[word] += 1

                word_index += 1

        return training_set, validation_set



