from importlib.util import find_spec

OPENAI_INSTALLED = find_spec("openai") is not None
REPLICATE_INSTALLED = find_spec("replicate") is not None
OLLAMA_INSTALLED = find_spec("ollama") is not None
