# Trivial tokenizers

`tokenizer/backend: trivial_numeric` uses `TrivialNumericTokenizer`: input is
space-separated integer IDs, and end of text is ID 0. The existing `trivial`
backend name remains an alias for this numeric format.

`tokenizer/backend: trivial_character` uses `TrivialCharacterTokenizer`. Each
character becomes its Unicode code point (`ord`), so ASCII characters retain
their ASCII IDs. There are no merges, inserted tokens, or normalization.

```yaml
tokenizer:
  backend: trivial_character
  character:
    charset: '0123456789+*=|'
    eot: '|'
```

The charset must be nonempty, contain unique characters, and include the
single end-of-text character. Unknown input characters raise `ValueError`.
The default charset is Python's `string.printable`, with `|` as end of text.
Decoding uses `chr`, including for model predictions outside the input charset.
Set the model vocabulary size to at least the largest code point in the charset
plus one; 128 covers ASCII. Charset order does not change token IDs.
