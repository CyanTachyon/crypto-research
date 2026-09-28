import yaml
from pathlib import Path
from crypto_transformer import CONFIGS_DIR


def load_config(config_name: str = "default") -> dict:
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
