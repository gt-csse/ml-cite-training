# ML-cite-training

ML-cite is an independent project developing machine learning methods for
legal citation extraction. Our companion project,
[mellea-lrc](https://github.com/gt-csse/mellea-lrc), checks citation identities
and whether cited sources support the associated claims.

This is the **training repository** for the natural
[ML-cite dataset](https://huggingface.co/datasets/gt-csse/ML-cite-training).
It fine-tunes BERT to extract case name, court, date/year and pin cite as
spans or absence.

Span annotations can be difficult to read as raw offsets. Browse them in the
[ML-cite visualization](https://mellea-lrc-visualizer.woodygoodenough.com/dataset?dataset=ml-cite).

The dataset is currently **private**, with public release planned soon.
Until then, your Hugging Face account needs dataset access. Once it is public,
the setup below will work without special access.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.11. From this directory, log in to Hugging Face first, then train:

```bash
uv run --locked hf auth login
uv run --locked train.py
```

Use a Hugging Face token with read access when prompted. `uv` installs the
dependencies automatically. The script downloads the training/development
data and saves the trained model and run records in `runs/bert/`.

To verify that a complete local dataset release's document annotations reproduce
its prepared train/dev/test files byte for byte:

```bash
uv run python tools/convert_training.py --release /path/to/ML-cite --verify
```
