"""Download only the CUAD benchmark and its referenced corpus files."""

import json
import re
import unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "awinml/legalbench-rag"
DATA_ROOT = Path("data/raw")
BENCHMARK = "benchmarks/cuad.json"
CORPUS_ROOT = "corpus/cuad"


def download_file(filename: str) -> Path:
    cached = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=DATA_ROOT,
    )
    return Path(cached)


def canonical_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized.encode("ascii", "ignore").decode())


def resolve_hub_path(relative_path: str, hub_files: set[str]) -> str:
    exact = f"{CORPUS_ROOT}/{relative_path.removeprefix('cuad/')}"
    if exact in hub_files:
        return exact
    target = canonical_name(Path(relative_path).name)
    matches = [
        path for path in hub_files
        if path.startswith(CORPUS_ROOT + "/")
        and canonical_name(Path(path).name) == target
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve {relative_path}: {matches}")
    return matches[0]


def main() -> None:
    benchmark_path = download_file(BENCHMARK)
    tests = json.loads(benchmark_path.read_text(encoding="utf-8"))["tests"]
    needed_paths = {
        snippet["file_path"]
        for test in tests
        for snippet in test["snippets"]
    }
    hub_files = set(list_repo_files(REPO_ID, repo_type="dataset"))
    for path in sorted(needed_paths):
        if not path.startswith("cuad/"):
            raise ValueError(f"Unexpected corpus path: {path}")
        source = resolve_hub_path(path, hub_files)
        downloaded = download_file(source)
        target = DATA_ROOT / "corpus" / CORPUS_ROOT.removeprefix("corpus/") / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != target.resolve():
            target.replace(downloaded) if target.exists() else downloaded.replace(target)
    print(f"Downloaded benchmark and {len(needed_paths)} corpus documents to {DATA_ROOT}")


if __name__ == "__main__":
    main()
