import pandas as pd
import os
import numpy as np

"""
Preprocessor to process text for the Neural Network.
Uses the skipgram approach (target -> context/neighborhood detection)
"""


class Preprocessor(object):
    def __init__(self, dimensions):
        self.corpus_file_path = os.path.join(os.getcwd(), "data", "data.txt")
        self.corpus = []
        self.vocabulary = []

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

    # remove . and ,
    def __remove_punc(self):
        cpr = []
        for text in self.corpus:
            tmp = text.replace(",", "")
            cpr.append(tmp.replace(".", ""))

        # update
        self.corpus = cpr

    def __clean(self):
        self.__remove_stop_words()
        self.__remove_punc()

    def __build_vocabulary(self):
        for text in self.corpus:
            tokens = text.split()
            for token in tokens:
                # add to vocabulary
                if token not in self.vocabulary:
                    self.vocabulary.append(token)

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
                for neighbor in s[max(idx - self.dimensions, 0): min(idx + self.dimensions, len(s) + 1)]:
                    if neighbor != word:
                        data.append([word, neighbor])

        columns = ["focus_word", "context"]
        df = pd.DataFrame(data, columns=columns)
        return df

    # one hot encode a single word
    def __one_hot_encode_word(self, word):
        one_hot = np.zeros(len(self.vocabulary), dtype=np.float32)
        word_idx_in_vocab = self.int_labels[word]
        one_hot[word_idx_in_vocab] = 1

        return one_hot

    # one hot encode all words in the generated data frame
    def __one_hot_encode_df(self, df):
        focus_words = []
        contexts = []

        for focus_word, context in zip(df["focus_word"], df["context"]):
            focus_words.append(self.__one_hot_encode_word(focus_word))

            # for context word
            encoded_context = self.__one_hot_encode_word(context)
            contexts.append(encoded_context)

        return np.array(focus_words, dtype=np.float32), np.array(contexts, dtype=np.float32)

    # runs all the steps sequentially
    def __pipeline(self):
        self.__read_corpus_from_file()
        self.__clean()
        self.__build_vocabulary()

        data_frame = self.__generate_data()
        focus_words, contexts = self.__one_hot_encode_df(df=data_frame)
        return data_frame, focus_words, contexts

    # the method to call!
    def run(self):
        return self.__pipeline()
