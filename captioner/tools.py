
from keras.layers import Embedding
from pathlib import Path

class Tools:

    def load_embedding(self, path=Path('.'), file=""):
        self.em = Embedding()

    def generate_embedded_input(self, input, vocab_size):
        test = 1