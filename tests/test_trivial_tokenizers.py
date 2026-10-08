import pytest

from theseus.data.tokenizer import (
    TokenizerConfig, TrivialCharacterTokenizer, TrivialNumericTokenizer, get_tokenizer,
)


def test_numeric_backend_compatibility():
    for backend in ('trivial', 'trivial_numeric'):
        tokenizer = get_tokenizer(TokenizerConfig(backend=backend))
        assert isinstance(tokenizer, TrivialNumericTokenizer)
        assert tokenizer.encode('1 73 0') == [1, 73, 0]
        assert tokenizer.decode([1, 73, 0]) == '1 73 0'
        assert tokenizer.eot_token == 0


def test_character_codepoints_and_roundtrip():
    tokenizer = get_tokenizer(TokenizerConfig(
        backend='trivial_character', charset='IVX+|é', eot_character='|',
    ))
    assert isinstance(tokenizer, TrivialCharacterTokenizer)
    assert tokenizer.encode('XIV|') == [88, 73, 86, 124]
    assert tokenizer.encode_ordinary('é') == [233]
    assert tokenizer.eot_token == 124
    texts = ['XIV|', '', 'é+I|']
    assert tokenizer.decode_batch(tokenizer.encode_batch(texts)) == texts
    assert tokenizer.decode([65]) == 'A'  # A model may predict outside the input charset.
    with pytest.raises(ValueError, match='outside charset'):
        tokenizer.encode('A')
    other = get_tokenizer(TokenizerConfig(backend='trivial_character', charset='A|'))
    assert other.encode('A|') == [65, 124]
    with pytest.raises(ValueError, match='outside charset'):
        other.encode('I')


@pytest.mark.parametrize('charset,eot', [('', '|'), ('II|', '|'), ('I', '|'), ('I|', 'II')])
def test_invalid_character_configuration(charset, eot):
    with pytest.raises(ValueError):
        TrivialCharacterTokenizer(charset, eot)
