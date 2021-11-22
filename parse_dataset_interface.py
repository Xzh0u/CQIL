# this file would parse the dataset in CodeNN into the form of CQIL
import tables
import json
import numpy as np

desc_path = "./data/origin/valid.desc.h5"
token_path = "./data/origin/valid.tokens.h5"
name_path = "./data/origin/valid.name.h5"
desc_vocab_path = "./data/origin/vocab.desc.json"
token_vocab_path = "./data/origin/vocab.tokens.json"
name_vocab_path = "./data/origin/vocab.name.json"
output_path = "./data/transformed_codenn/valid.json"


def load_vocab(path):
    vocab = json.load(open(path, 'r'))
    return vocab


def load_data(path):
    with tables.open_file(path) as f:
        phrases = f.get_node('/phrases')[:].astype(np.long)
        indices = f.get_node('/indices')[:]
    return phrases, indices


def transform_data(vocab, contents, idxs):
    # transform idxs data to word data (i.e. transform (0, 1) to ('<pad>', '<s>'))
    vocab = dict(zip(vocab.values(), vocab.keys()))
    data = []
    for idx in idxs:
        raw_idx_data = contents[idx[1]:idx[1]+idx[0]]
        raw_data = [vocab[i] for i in raw_idx_data]
        data.append(' '.join(raw_data))
    return data

def main():
    desc_vocab = load_vocab(desc_vocab_path)
    descs, desc_idxs = load_data(desc_path)
    transformed_desc = transform_data(desc_vocab, descs, desc_idxs)

    token_vocab = load_vocab(token_vocab_path)
    tokens, token_idxs = load_data(token_path)
    transformed_token = transform_data(token_vocab, tokens, token_idxs)

    name_vocab = load_vocab(name_vocab_path)
    names, name_idxs = load_data(name_path)
    transformed_name = transform_data(name_vocab, names, name_idxs)

    output = []
    for desc, token, name in zip(transformed_desc, transformed_token, transformed_name):
        output.append({
            'query': desc,
            'name': name,
            'body': token
        })

    with open(output_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()

