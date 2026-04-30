from functools import lru_cache

import tiktoken

_FALLBACK_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return the number of tokens in *text* for the given OpenAI model."""
    return len(_get_encoding(model).encode(text))
