from torch.utils.data import Dataset
import torch
class TinyShakespeare(Dataset):
    def __init__(self, data_root, context_len, split):
        self.data_root = data_root
        self.context_length = context_len
        self.split = split
        with open(self.data_root, "r", encoding="utf-8") as txt_file:
            self.text = txt_file.read()
            # Here are all the unique characters that occur in this text.
            self.charset = sorted(list(set(self.text)))
            self.vocab_size = len(self.charset) # Number of possible elements (tokens) in training sequence.
        if self.split == "train":
            self.text = self.text[:int(len(self.text) * 0.8)]
        if self.split == "val":
            self.text = self.text[int(len(self.text) * 0.8):]
        # Tokenization on character level.
        stoi = {element:idx for idx, element in enumerate(self.charset)}
        itos = {idx:element for idx, element in enumerate(self.charset)}
        encode = lambda s: [stoi[c] for c in s] # Encoder: take a string, output a list of integers.
        decode = lambda i: ''.join([itos[idx] for idx in i]) # Decoder: take a list of index (tokenized output element), output a string.
        self.text = encode(self.text)
        # Remember in tokenization, you basically trade off the vocabulary size and sequence length. So you can have very long sequence
        # with a relatively small vocabulary size as well as you can have a very short sequence with bigger vocabulary.For this reason
        # in between character-level and word-level language modeling, currently SOTA models use "sub-word" level language modeling by
        # for example Sentence Piece or BPE (in GPT).
    def __len__(self):
        return len(self.text) - self.context_length - 1
        # Every token in the sequence (in this case every character) is a starting sequence of a sample except for the last context sequence
        # because the last sequence of tokens can only be the target sequence of input sequence starting at position (text_len - context_len - 1).
    def __getitem__(self, index):
        # When you sample a chunk of data (pack a sequence of tokens within a context) is actually has multiple examples packed into it.
        # Thats because all of the characters sequentially follow each other. For example in a chunk of token sequence with length 9,
        # there's actually 8 individual examples packed in. Each token of a sequence (in the 9 sequence chunk) is input the model and
        # it is expected to predict the next token of the sequence by the model. Therefore, each output of the model's predictions are
        # the next input's predictors. Because while we are sampling, we can start the sampling generation as little as one token of
        # context and the Transformer knows how to predict the next character with all the way up to just context of one and so then it
        # can predict everything up to block size (context length) and after block size we have to start truncating. Because Transformer
        # will never receive more than block size inputs when its predicting the next character.
        input_sample = torch.tensor(self.text[index:index+self.context_length], dtype=torch.long)
        target_sample = torch.tensor(self.text[index + 1: index + 1 + self.context_length], dtype = torch.long)

        return input_sample, target_sample
# Test
if __name__ == "__main__":
    tshakespeare_dataset = TinyShakespeare("input.txt", 256, "val")
    input_sample, target_sample = tshakespeare_dataset.__getitem__(5)
    itos = {idx: element for idx, element in enumerate(tshakespeare_dataset.charset)}

    print(f"Input Sample -> {''.join(itos[int(idx)] for idx in input_sample)}")
    print(f"Target Sample -> {''.join(itos[int(idx)] for idx in target_sample)}")