from tokenizers import ByteLevelBPETokenizer


class Tokenizer:
    def __init__(self, data, vocab_size=None, min_frequency=None, special_tokens=None,padding_size=None):
        self.char_token = None
        self.data = data
        self.vocab_size = vocab_size if vocab_size is not None else 1000
        self.min_frequency = min_frequency if min_frequency is not None else 2
        self.special_tokens = special_tokens if special_tokens is not None else ["<s>", "<pad>", "</s>", "<unk>","<mask>"]
        self.padding_size=padding_size if padding_size is not None else 300
        self.character_tokenizer()

    def character_tokenizer(self):
        token = self._get_char_tokenizer()
        if isinstance(self.data, list):
            token.train_from_iterator(self.data, vocab_size=self.vocab_size,
                                      min_frequency=self.min_frequency, special_tokens=self.special_tokens)
        elif isinstance(self.data, str):
            token.train(files=[self.data], vocab_size=self.vocab_size,
                        min_frequency=self.min_frequency, special_tokens=self.special_tokens)
        padding_token = "[PAD]"
        token.add_special_tokens([padding_token])
        self.char_token = token

    def encode(self,data):
        tokenized_batch = self.char_token.encode_batch(data)
        for tokens in tokenized_batch:
            tokens.pad(self.padding_size)
        return tokenized_batch

    def decode(self,data):
        return self.char_token.decode(data)

    def serialize_tokenizer(self):
        self.char_token.save_model("C:\\Toothless\\tokenizer")

    def load_tokenizer(self, path,merge):
        self.char_token = ByteLevelBPETokenizer.from_file(path,merge)

    def _get_char_tokenizer(self):
        return ByteLevelBPETokenizer()
