"""Step 4: one-command automation of the whole lifecycle (the "CI/CD in Python" part).

Run:  python pipeline.py            # preprocess -> train -> test -> quality gate -> monitor
      python pipeline.py --watch    # re-run automatically whenever data or code changes

The same stages run in GitHub Actions (.github/workflows/ci.yml) on every push.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time

from src import config

WATCHED = [config.RAW_DATA, *sorted((config.ROOT / "src").glob("*.py"))]


def run(cmd: list[str], stage: str) -> None:
    print(f"\n=== {stage} ===")
    if subprocess.run(cmd, cwd=config.ROOT).returncode != 0:
        print(f"Stage '{stage}' failed - stopping pipeline.")
        sys.exit(1)


def quality_gate() -> None:
    """Only allow deployment if the new model beats the minimum ROC-AUC."""
    auc = json.loads(config.METRICS_PATH.read_text())["roc_auc"]
    print(f"\n=== Quality gate ===\nROC-AUC {auc} (minimum {config.MIN_ROC_AUC_TO_DEPLOY})")
    if auc < config.MIN_ROC_AUC_TO_DEPLOY:
        print("Model rejected: not good enough to deploy.")
        sys.exit(1)
    print("Model approved for deployment.")


def tag_release() -> None:
    """Use GitPython to tag the approved model version (skipped if not a git repo)."""
    try:
        import git
        repo = git.Repo(config.ROOT)
        auc = json.loads(config.METRICS_PATH.read_text())["roc_auc"]
        tag = f"model-{time.strftime('%Y%m%d-%H%M%S')}-auc{auc}"
        repo.create_tag(tag)
        print(f"Created git tag: {tag}")
    except Exception as exc:  # not a repo, no commits yet, etc.
        print(f"(Skipping git tag: {exc.__class__.__name__})")


def run_pipeline() -> None:
    py = sys.executable
    run([py, "-m", "src.download_data"], "0. Get data (Google Drive)")
    run([py, "-m", "src.preprocess"], "1. Preprocess")
    run([py, "-m", "src.train"], "2. Train + log to MLflow")
    run([py, "-m", "pytest", "-q", "tests"], "3. Automated tests")
    quality_gate()
    run([py, "-m", "monitoring.monitor"], "4. Monitoring report")
    tag_release()
    print("\nPipeline finished. Deploy with:  uvicorn api.main:app")


def fingerprint() -> str:
    h = hashlib.md5()
    for p in WATCHED:
        h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="re-run when data/code changes")
    args = parser.parse_args()
    run_pipeline()
    if args.watch:
        last = fingerprint()
        print("\nWatching for changes (Ctrl+C to stop)...")
        while True:
            time.sleep(5)
            now = fingerprint()
            if now != last:
                print("\nChange detected - re-running pipeline.")
                last = now
                run_pipeline()
