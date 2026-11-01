from pathlib import Path

from src.eval import _load_config


def test_baseline_and_final_configs_load():
    for name in ("baseline.yaml", "final.yaml"):
        config = _load_config(Path("configs") / name)
        assert config["chunking"]["strategy"] == "fixed"
        assert config["retrieval"]["method"] == "bm25"
        assert config["retrieval"]["top_k"] == 8
