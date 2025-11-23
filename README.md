# Two-Tower Retrieval (Dual Encoder)

Production-style dual encoder for large-scale retrieval, inspired by Pinterest’s two-tower setup. The system learns user and item representations separately and scores them via dot-product similarity to power candidate generation.

## What’s Inside
- User & item towers in PyTorch with configurable MLP heads
- Contrastive / in-batch negative loss for scalable retrieval
- ANN-ready embedding export with a lightweight demo index
- Reproducible configs, Makefile, Dockerfile, and CI scaffold
- Teaching-style comments and structure for learners and reviewers

## Who This Is For
ML engineers who want a clean baseline for retrieval systems: reproducible training, clear configs, and production-minded layout (separation of data/model/train/eval/serve).

## Architecture
- **User Tower:** embeds IDs and (later) recency/behavioral aggregates
- **Item Tower:** embeds IDs and (later) content features (text/title/category/image)
- **Scoring:** normalized dot product between user and item embeddings
- **Loss:** InfoNCE-style in-batch negative sampling
- **Retrieval:** ANN index (FAISS/ScaNN ready) for low-latency candidate generation

## Repo Layout
- `configs/` — `model.yaml`, `train.yaml`, `data.yaml` for Hydra/YAML-driven runs
- `src/two_tower/` — package code
  - `models/` — towers and scoring
  - `training/` — training loop and loss
  - `eval/` — recall@K / MRR utilities
  - `serve/` — ANN index stub and API entrypoint
  - `features/` — feature plumbing (expandable)
  - `utils/` — logging, helpers
- `scripts/` — CLI entrypoints (`train.py`, `eval.py`, `serve.py`)
- `tests/` — unit tests (shapes/metrics)
- `data/` — empty; see `data/README.md` for schema expectations
- `.github/workflows/` — CI for lint + tests
- `Dockerfile` — slim runtime image
- `Makefile` — common tasks (setup, lint, test, train, serve)

## Quickstart
- `make setup` — create venv and install runtime deps
- `make install-dev` — add lint/type/test tooling
- `make lint && make test` — static + unit checks
- `make train` — run a dummy training loop (placeholder until data is wired)
- `make serve` — run ANN demo

## Roadmap
- Add real dataloaders & feature builders (IDs + optional text/image features)
- Integrate FAISS/ScaNN-backed ANN index build/load with export scripts
- Add hard-negative mining and richer logging/monitoring
- Provide notebook walkthroughs and end-to-end example runs

## License
MIT — see `LICENSE`.
