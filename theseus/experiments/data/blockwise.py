"""Tokenization jobs for finite, indexable datasets."""

from theseus.data import datasets
from theseus.data.tokenize import TokenizeBlockwiseDatasetJob
from theseus.registry import job


@job("data/tokenize/alpaca")
class TokenizeAlpaca(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.Alpaca


@job("data/tokenize/bbq")
class TokenizeBBQ(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.BBQ


@job("data/tokenize/cfq")
class TokenizeCFQ(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.CFQ


@job("data/tokenize/clutrr")
class TokenizeCLUTRR(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.CLUTRR


@job("data/tokenize/fever")
class TokenizeFEVER(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.FEVER


@job("data/tokenize/flan")
class TokenizeFlan(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.Flan


@job("data/tokenize/harmfulqa")
class TokenizeHarmfulQA(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.HarmfulQA


@job("data/tokenize/longbench")
class TokenizeLongBench(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.LongBench


@job("data/tokenize/longhealth")
class TokenizeLongHealth(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.LongHealth


@job("data/tokenize/mmlu")
class TokenizeMMLU(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.MMLU


@job("data/tokenize/mnli")
class TokenizeMNLI(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.MNLI


@job("data/tokenize/mtob")
class TokenizeMTOB(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.MTOB


@job("data/tokenize/openr1_math")
class TokenizeOpenR1Math(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.OpenR1Math


@job("data/tokenize/qqp")
class TokenizeQQP(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.QQP


@job("data/tokenize/siqa")
class TokenizeSIQA(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.SIQA


@job("data/tokenize/squad")
class TokenizeSQuAD(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.SQuAD


@job("data/tokenize/sst2")
class TokenizeSST2(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.SST2


@job("data/tokenize/winogrande")
class TokenizeWinogrande(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.Winogrande


@job("data/tokenize/addition")
class TokenizeAddition(TokenizeBlockwiseDatasetJob):
    DATASET = datasets.Addition
