import os
from huggingface_hub import HfApi

api = HfApi(token="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")
api.upload_folder(
    folder_path="pokemon_gpt2xl",
    repo_id="gelan32/pokemon-gp2",
    repo_type="model",
)
