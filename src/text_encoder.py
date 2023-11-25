import os
import json
import regex as re
from functools import lru_cache

@lru_cache()
def unicode_to_bytes():
    """
    Returns a dictionary mapping unicode strings to corresponding utf-8 byte lists.
    This function is essential for working with reversible bpe codes on unicode strings.
    """
    unicode_chars = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    utf8_bytes = unicode_chars[:]
    n = 0
    for b in range(2**8):
        if b not in unicode_chars:
            unicode_chars.append(b)
            utf8_bytes.append(2**8 + n)
            n += 1
    utf8_bytes = [bytes([n]) for n in utf8_bytes]
    return dict(zip(unicode_chars, utf8_bytes))

def symbol_pairs(word):
    """Returns a set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class TextEncoder:
    def __init__(self, char_encoder, bpe_merges, errors='replace'):
        self.char_encoder = char_encoder
        self.char_decoder = {v: k for k, v in self.char_encoder.items()}
        self.errors = errors
        self.unicode_encoder = unicode_to_bytes()
        self.unicode_decoder = {v: k for k, v in self.unicode_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Added re.IGNORECASE for BPE merges to happen for capitalized contractions
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def apply_bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = symbol_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = symbol_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode_text(self, text):
        bpe_tokens = []
        for token in re.findall(self.pattern, text):
            token = ''.join(self.unicode_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.char_encoder[bpe_token] for bpe_token in self.apply_bpe(token).split(' '))
        return bpe_tokens

    def decode_tokens(self, tokens):
        text = ''.join([self.char_decoder[token] for token in tokens])
        text = bytearray([self.unicode_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def load_text_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'char_encoder.json'), 'r') as f:
        char_encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'bpe_vocab.txt'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return TextEncoder(
        char_encoder=char_encoder,
        bpe_merges=bpe_merges,
    )
