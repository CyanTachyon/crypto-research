from crypto_transformer.utils.config import load_config
from crypto_transformer import PROJECT_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR


def test_config_loads():
    config = load_config()
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert config["model"]["num_classes"] == 3


def test_paths_exist():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert RAW_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert CHECKPOINTS_DIR.exists()
