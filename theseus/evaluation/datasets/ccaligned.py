"""
CCAligned perplexity evaluation.

Measures language model perplexity on multilingual parallel text.
Returns 1/perplexity (higher is better).

Provides per-language eval classes so each can be tracked independently
across continual learning phases.  All ~90 CCAligned language pairs are
registered as ``ccaligned_<lang>_ppl`` (e.g. ``ccaligned_fr_xx_ppl``).
"""

from dataclasses import dataclass
from typing import cast

from omegaconf import DictConfig

from theseus.config import build, configuration, configure, field
from theseus.data.datasets.ccaligned import CCAligned
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@dataclass
class CCAlignedEvalConfig:
    num_samples: int = field("eval/ccaligned/num_samples", default=500)


# All language codes available on statmt.org/cc-aligned/sentence-aligned/
CCALIGNED_LANGS = [
    "af_ZA",
    "am_ET",
    "ar_AR",
    "as_IN",
    "az_AZ",
    "be_BY",
    "bg_BG",
    "bm_ML",
    "bn_IN",
    "br_FR",
    "bs_BA",
    "ca_ES",
    "cb_IQ",
    "cs_CZ",
    "cx_PH",
    "cy_GB",
    "da_DK",
    "de_DE",
    "el_GR",
    "es_XX",
    "et_EE",
    "fa_IR",
    "ff_NG",
    "fi_FI",
    "fr_XX",
    "ga_IE",
    "gl_ES",
    "gu_IN",
    "ha_NG",
    "he_IL",
    "hi_IN",
    "hi_IN_rom",
    "hr_HR",
    "ht_HT",
    "hu_HU",
    "hy_AM",
    "id_ID",
    "ig_NG",
    "is_IS",
    "it_IT",
    "ja_XX",
    "jv_ID",
    "ka_GE",
    "kg_AO",
    "kk_KZ",
    "km_KH",
    "kn_IN",
    "ko_KR",
    "ku_TR",
    "ky_KG",
    "lg_UG",
    "ln_CD",
    "lo_LA",
    "lt_LT",
    "lv_LV",
    "mg_MG",
    "mi_NZ",
    "mk_MK",
    "ml_IN",
    "mn_MN",
    "mr_IN",
    "ms_MY",
    "mt_MT",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "no_XX",
    "ns_ZA",
    "ny_MW",
    "om_KE",
    "or_IN",
    "pa_IN",
    "pl_PL",
    "ps_AF",
    "pt_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "sk_SK",
    "sl_SI",
    "sn_ZW",
    "so_SO",
    "sq_AL",
    "sr_RS",
    "ss_SZ",
    "st_ZA",
    "su_ID",
    "sv_SE",
    "sw_KE",
    "ta_IN",
    "ta_IN_rom",
    "te_IN",
    "te_IN_rom",
    "tg_TJ",
    "th_TH",
    "ti_ET",
    "tl_XX",
    "tn_BW",
    "tr_TR",
    "ts_ZA",
    "uk_UA",
    "ur_PK",
    "ur_PK_rom",
    "ve_ZA",
    "vi_VN",
    "wo_SN",
    "xh_ZA",
    "yo_NG",
    "zh_CN",
    "zh_TW",
    "zu_ZA",
]


class CCAlignedEval(PerplexityEvaluation):
    """Perplexity evaluation on CCAligned multilingual sentence pairs."""

    _lang: str = "fr_XX"
    CONFIG = CCAlignedEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        config = cast(DictConfig, build(CCAligned.CONFIG))
        config.data.ccaligned.language = self._lang
        with configuration(config):
            ds = CCAligned()
        self.items: list[str] = []
        for text in ds:
            self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return f"ccaligned_{self._lang.lower()}_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]


def _make_lang_eval(lang: str) -> type[CCAlignedEval]:
    """Create a CCAlignedEval subclass for a specific language."""
    cls_name = f"CCAligned{''.join(p.capitalize() for p in lang.split('_'))}Eval"
    return type(cls_name, (CCAlignedEval,), {"_lang": lang})


# Build a dict of lang_key -> eval class for all languages.
# Registry keys are like "ccaligned_fr_xx_ppl", "ccaligned_de_de_ppl", etc.
CCALIGNED_EVALS: dict[str, type[CCAlignedEval]] = {
    f"ccaligned_{lang.lower()}_ppl": _make_lang_eval(lang) for lang in CCALIGNED_LANGS
}

# Short aliases (e.g. "ccaligned_fr_ppl" -> CCAligned fr_XX) so eval names line up
# with the on-disk tokenized suffix used by the training-side data configs
# (configs/data/cl100k/ccaligned_{fr,de,zh}.yaml use suffix 'fr'/'de'/'zh').
CCALIGNED_SHORT_ALIASES: dict[str, type[CCAlignedEval]] = {
    f"ccaligned_{lang.split('_')[0].lower()}_ppl": cls
    for lang, cls in (
        (lang, CCALIGNED_EVALS[f"ccaligned_{lang.lower()}_ppl"])
        for lang in CCALIGNED_LANGS
    )
}

# Route every registration through the ``evaluation`` decorator so the
# ``_ppl`` suffix guard fires on these too.
for _key, _cls in {**CCALIGNED_EVALS, **CCALIGNED_SHORT_ALIASES}.items():
    evaluation(_key)(_cls)
