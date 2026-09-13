"""Tokenization jobs for streaming pretraining datasets."""

from theseus.data import datasets
from theseus.data.tokenize import TokenizeVariableDatasetJob
from theseus.registry import job


@job("data/tokenize/ccaligned")
class TokenizeCCAligned(TokenizeVariableDatasetJob):
    DATASET = datasets.CCAligned


@job("data/tokenize/fineweb")
class TokenizeFineWeb(TokenizeVariableDatasetJob):
    DATASET = datasets.FineWeb


@job("data/tokenize/pes2o")
class TokenizePes2O(TokenizeVariableDatasetJob):
    DATASET = datasets.Pes2O


@job("data/tokenize/pg19")
class TokenizePG19(TokenizeVariableDatasetJob):
    DATASET = datasets.PG19


@job("data/tokenize/pile")
class TokenizePile(TokenizeVariableDatasetJob):
    DATASET = datasets.Pile


@job("data/tokenize/pile_detoxify")
class TokenizePileDetoxify(TokenizeVariableDatasetJob):
    DATASET = datasets.PileDetoxify


@job("data/tokenize/pile_injected")
class TokenizePileInjected(TokenizeVariableDatasetJob):
    DATASET = datasets.PileInjected
