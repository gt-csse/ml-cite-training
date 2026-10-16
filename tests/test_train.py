import copy
import json

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

import train


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Alpha", "v", ".",
        "Beta", ",", "1", "F", "3", "##d", "2", "4", "(", "X", "2000", ")",
        "<", ">", "/", "target", "_", "locator",
    ]) + "\n")
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)


def row(row_id="train-1"):
    context = "Alpha v. Beta, 1 F.3d 2, 4 (X 2000)."

    def span(quote):
        start = context.index(quote)
        return {"start": start, "end": start + len(quote), "quote": quote}

    return {"id": row_id, "context_text": context, "locator": span("1 F.3d 2"),
            "targets": dict(zip(train.FIELDS, map(span, ["Alpha v. Beta", "X", "2000", "4"])))}


def test_encoding_is_gold_independent_and_exact(tokenizer):
    record = row()
    example = train.encode(tokenizer, record, 128)
    assert tokenizer.convert_ids_to_tokens(example.input_ids[example.positions[0]]) == "Alpha"
    first, last = example.targets[0]
    assert first == 0 and tokenizer.convert_ids_to_tokens(
        example.input_ids[example.positions[last]]) == "Beta"
    null_record = copy.deepcopy(record)
    null_record["targets"] = dict.fromkeys(train.FIELDS)
    null_example = train.encode(tokenizer, null_record, 128)
    assert null_example.input_ids == example.input_ids
    assert null_example.positions == example.positions
    assert null_example.targets == [None] * 4


def test_invalid_source_is_not_a_representation_exclusion(tokenizer):
    record = row()
    record["targets"]["court"]["quote"] = "wrong"
    with pytest.raises(ValueError, match="invalid offsets or source quote") as error:
        train.encode(tokenizer, record, 128)
    assert not isinstance(error.value, train.RepresentationError)
    record = row()
    del record["targets"]["date"]
    with pytest.raises(ValueError, match="four explicit targets"):
        train.encode(tokenizer, record, 128)


def test_unrepresentable_spans_fail_explicitly(tokenizer):
    record = row()
    record["targets"]["case_name"] = {"start": 0, "end": 2, "quote": "Al"}
    with pytest.raises(train.RepresentationError, match="exact span"):
        train.encode(tokenizer, record, 128)
    record = row()
    end = record["locator"]["end"]
    record["targets"]["case_name"] = {"start": 0, "end": end,
                                         "quote": record["context_text"][:end]}
    with pytest.raises(train.RepresentationError, match="locator markers"):
        train.encode(tokenizer, record, 128)
    with pytest.raises(train.RepresentationError, match="no truncation"):
        train.encode(tokenizer, row(), 4)


def test_null_loss_trains_presence_without_pointer_supervision():
    outputs = [tuple(torch.randn(3, requires_grad=True) for _ in range(3)) for _ in train.FIELDS]
    train.span_loss(outputs, [None] * 4).backward()
    for start, end, presence in outputs:
        assert start.grad is None and end.grad is None
        assert presence.grad is not None and torch.isfinite(presence.grad).all()


def test_one_epoch_download_train_save_and_reload(tmp_path, tokenizer, monkeypatch):
    snapshot = tmp_path / "snapshot"
    (snapshot / "data").mkdir(parents=True)
    valid = row()
    absent = row("train-null")
    absent["targets"] = dict.fromkeys(train.FIELDS)
    excluded = row("train-unrepresentable")
    excluded["targets"]["case_name"] = {"start": 0, "end": 2, "quote": "Al"}
    for split, rows in [("train", [valid, absent, excluded]), ("dev", [row("dev-1")])]:
        (snapshot / "data" / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in rows))
    downloads = []

    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return str(snapshot)

    config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)
    initial = {}

    def model_loader(*args, **kwargs):
        model = BertModel(config, add_pooling_layer=False)
        initial["embeddings"] = model.embeddings.word_embeddings.weight.detach().clone()
        assert all(parameter.requires_grad for parameter in model.parameters())
        return model

    reload_model = BertModel.from_pretrained
    monkeypatch.setattr(train, "snapshot_download", download)
    monkeypatch.setattr(train.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(train.BertTokenizerFast, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(train.BertModel, "from_pretrained", model_loader)
    output = tmp_path / "run"
    train.main(["--output-dir", str(output), "--epochs", "1", "--batch-size", "1",
                "--grad-accum", "2", "--learning-rate", "0.001", "--device", "cpu"])
    assert downloads == [(("gt-csse/ML-cite-training",), {
        "repo_type": "dataset", "revision": "main",
        "allow_patterns": ["data/train.jsonl", "data/dev.jsonl"],
    })]
    metadata = json.loads((output / "training.json").read_text())
    assert metadata["status"] == "complete" and metadata["selected_epoch"] == 1
    assert metadata["rows"] == {"train": 2, "dev": 1}
    assert len(json.loads((output / "exclusions.json").read_text())) == 1
    assert json.loads((output / "exclusions.json").read_text())[0]["id"] == "train-unrepresentable"
    monkeypatch.undo()
    encoder = reload_model(output / "bert", add_pooling_layer=False, local_files_only=True)
    assert not torch.equal(initial["embeddings"], encoder.embeddings.word_embeddings.weight)
    heads = train.SpanHeads(config.hidden_size)
    heads.load_state_dict(torch.load(output / "heads.pt", weights_only=True))
    example = train.encode(tokenizer, valid, 128)
    outputs = train.batch_outputs(encoder, heads, [example], torch.device("cpu"), tokenizer.pad_token_id)
    assert len(outputs[0]) == 4 and torch.isfinite(train.span_loss(outputs[0], example.targets))
    assert (output / "tokenizer" / "tokenizer.json").is_file()
