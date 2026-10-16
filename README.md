# ML-cite-training

Download the natural [ML-cite dataset](https://huggingface.co/datasets/gt-csse/ML-cite-training)
and fine-tune `google-bert/bert-base-cased` to extract case name, court,
date/year and pin cite. Each field predicts a span or absence.

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run train.py
```

The script downloads the natural train/development files, marks the supplied
locator in each context, and trains the full BERT encoder with four span/presence
heads. It keeps the checkpoint with the highest development accuracy across
all four fields, counting correct absences. The test split is unused.

Weights, tokenizer, settings and training history are saved in `runs/bert/`.
Rows that exceed BERT's limits or cannot map exactly to its token spans are
listed in `runs/bert/exclusions.json`; labels are never rewritten as absent.

```bash
uv run train.py --epochs 4 --batch-size 8 --output-dir runs/bert-4epochs
uv run train.py --help
uv run pytest
```

The dataset and model revisions can be pinned with `--dataset-revision` and
`--model-revision`. CUDA is used when available; otherwise training uses CPU.
Advanced code and previous experiment records are local in `experiments/`,
which is ignored by Git.
