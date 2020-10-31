import pandas as pd


class Preprocessor(object):
    def __init__(self, dimensions):
        self.corpus_file_path = "data/data.txt"
        self.corpus = []
        self.vocabulary = set()

        self.dimensions = dimensions

        self.int_labels = dict()

    def __read_corpus_from_file(self):
        with open(self.corpus_file_path) as file:
            lines = file.readlines()

            for line in lines:
                self.corpus.append(line.strip().lower())

    def __remove_stop_words(self):
        stop_words = [
            "is", "an", "is", "the", "of", "from",
            "and", "a", "that", "for"
        ]

        cleaned_corpus = []

        for text in self.corpus:
            tokens = text.split()
            for sw in stop_words:
                if sw in tokens:
                    tokens.remove(sw)
            cleaned_corpus.append(" ".join(tokens))

        # update
        self.corpus = cleaned_corpus

    def __remove_period(self):
        cpr = []
        for text in self.corpus:
            cpr.append(text.replace(".", ""))

        # update
        self.corpus = cpr

    def __clean(self):
        self.__remove_stop_words()
        self.__remove_period()

    def __build_vocabulary(self):
        for text in self.corpus:
            tokens = text.split()
            for token in tokens:
                # add to vocabulary
                self.vocabulary.add(token)

    def __generate_data(self):
        # assign the index to word as an int label
        for idx, word in enumerate(self.vocabulary):
            self.int_labels[word] = idx

        tokenized_sentences = []
        for text in self.corpus:
            tokenized_sentences.append(text.split())

        data = []
        for s in tokenized_sentences:
            for idx, word in enumerate(s):
                # find neighboring words / context based on the dimensions (how many words to relate to)
                context = []
                for neighbor in s[max(idx - self.dimensions, 0): min(idx + self.dimensions, len(s) + 1)]:
                    if neighbor != word:
                        context.append(neighbor)
                data.append([word, context])

        columns = ["focus_word", "context"]
        df = pd.DataFrame(data, columns=columns)
        return df

    def __pipeline(self):
        self.__read_corpus_from_file()
        self.__clean()
        self.__build_vocabulary()

        data_frame = self.__generate_data()
        return data_frame

    def run(self):
        return self.__pipeline()
