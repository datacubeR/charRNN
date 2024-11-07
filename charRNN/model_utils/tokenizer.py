class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.n_vocabulary = len(vocabulary)

    def text_to_seq(self, text):
        seq = []
        for letter in text:
            seq.append(self.vocabulary.index(letter))

        return seq

    def seq_to_text(self, seq):
        text = ""
        for idx in seq:
            text += self.vocabulary[idx]

        return text
