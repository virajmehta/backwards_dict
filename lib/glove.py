
import json
import numpy as np

VOCAB_FILE_PATH ="lib/vocabulary.txt"
DEFAULT_FILE_PATH = "word_vectors/glove.6B.50d.txt"

def loadWordVectors(vec_file_path=DEFAULT_FILE_PATH,
                    tok_file_path=VOCAB_FILE_PATH,dimensions=50):
    """Read pretrained GloVe vectors"""
    f = open(VOCAB_FILE_PATH, 'r')
    tokens = json.load(f)
    f.close()

    wordVectors = np.zeros((len(tokens), dimensions))
    with open(vec_file_path) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors, tokens
