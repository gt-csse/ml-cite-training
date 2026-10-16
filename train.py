"""Download the natural ML-cite corpus and fine-tune BERT's four-field heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, BertModel, BertTokenizerFast

FIELDS = ("case_name", "court", "date", "pin_cite")
OPEN_LOCATOR = "\n<target_locator>\n"
CLOSE_LOCATOR = "\n</target_locator>\n"
MAX_SPAN_TOKENS = 64
PRESENCE_THRESHOLD = 0.5  # Raw logit, matching the existing BERT task.


class RepresentationError(ValueError):
    """Valid source annotation that this token-span model cannot represent."""


@dataclass
class Example:
    input_ids: list[int]
    positions: list[int]
    targets: list[tuple[int, int] | None]


def trim_span(context: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (context[start].isspace() or context[start] in "-."):
        start += 1
    while end > start and context[end - 1] in " ,\t\r\n":
        end -= 1
    return start, end


def encode(tokenizer, record: dict, max_tokens: int) -> Example:
    """Keep gold out of the input; require exact character-to-token alignment."""
    context = record.get("context_text")
    targets = record.get("targets")
    if not isinstance(context, str) or not isinstance(targets, dict) or set(targets) != set(FIELDS):
        raise ValueError("context_text and exactly four explicit targets are required")
    if "case_name" in record and record["case_name"] != targets["case_name"]:
        raise ValueError("case_name alias disagrees with targets.case_name")
    for field, span in [("locator", record.get("locator")), *targets.items()]:
        if span is None and field != "locator":
            continue
        if (not isinstance(span, dict)
                or type(span.get("start")) is not int or type(span.get("end")) is not int
                or not 0 <= span["start"] < span["end"] <= len(context)
                or context[span["start"]:span["end"]] != span.get("quote")):
            raise ValueError(f"{field}: invalid offsets or source quote")
    if not tokenizer.is_fast or any(x is None for x in
            (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)):
        raise ValueError("a fast BERT tokenizer with native CLS/SEP/PAD is required")
    left, right = record["locator"]["start"], record["locator"]["end"]
    ids, offsets = [tokenizer.cls_token_id], [None]
    for text, base in [(context[:left], 0), (OPEN_LOCATOR, None),
                       (context[left:right], left), (CLOSE_LOCATOR, None),
                       (context[right:], right)]:
        part = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids.extend(part["input_ids"])
        offsets.extend((base + a, base + b) if base is not None else None
                       for a, b in part["offset_mapping"])
    ids.append(tokenizer.sep_token_id)
    offsets.append(None)
    if len(ids) > max_tokens:
        raise RepresentationError(f"input has {len(ids)} tokens; limit is {max_tokens}; no truncation applied")
    positions = [i for i, offset in enumerate(offsets) if offset is not None and offset[0] < offset[1]]
    if not positions:
        raise ValueError("no source tokens")
    source_offsets = [offsets[i] for i in positions]
    aligned = []
    for field in FIELDS:
        target = targets[field]
        if target is None:
            aligned.append(None)
            continue
        start, end = target["start"], target["end"]
        first = next((i for i, (a, b) in enumerate(source_offsets) if a <= start < b), None)
        last = next((i for i, (a, b) in enumerate(source_offsets) if a < end <= b), None)
        if first is None or last is None or first > last:
            raise RepresentationError(f"{field}: boundaries cannot map to tokens")
        if last - first + 1 > MAX_SPAN_TOKENS:
            raise RepresentationError(f"{field}: span exceeds {MAX_SPAN_TOKENS} tokens")
        mapped = positions[first:last + 1]
        if mapped != list(range(mapped[0], mapped[-1] + 1)):
            raise RepresentationError(f"{field}: span crosses inserted locator markers")
        if trim_span(context, source_offsets[first][0], source_offsets[last][1]) != (start, end):
            raise RepresentationError(f"{field}: token envelope cannot reproduce exact span")
        aligned.append((first, last))
    return Example(ids, positions, aligned)


def load_examples(tokenizer, snapshot: Path, max_tokens: int):
    examples, exclusions, hashes, seen = {}, [], {}, set()
    for split, filename in [("train", "train.jsonl"), ("dev", "dev.jsonl")]:
        raw = (snapshot / "data" / filename).read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        examples[split] = []
        for row in rows:
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id in seen:
                raise ValueError(f"{split}: missing or duplicate row ID {row_id!r}")
            seen.add(row_id)
            try:
                examples[split].append(encode(tokenizer, row, max_tokens))
            except RepresentationError as error:
                exclusions.append({"split": split, "id": row_id, "reason": str(error)})
            except ValueError as error:
                raise ValueError(f"{split}:{row_id}: {error}") from error
        if not examples[split]:
            raise ValueError(f"{split}: no representable rows")
        print(f"{split}: {len(examples[split])}/{len(rows)} rows; "
              f"{len(rows) - len(examples[split])} explicit representation exclusions", flush=True)
    return examples, exclusions, hashes


class SpanHeads(nn.Module):
    """Four bilinear start/end/presence heads, queried by native CLS."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.heads = nn.ModuleList(nn.ModuleDict({
            f"{kind}_{boundary}": nn.Linear(hidden_size, rank, bias=False)
            for kind in ("query", "key") for boundary in ("start", "end", "presence")
        }) for _ in FIELDS)
        self.scale = 1 / math.sqrt(rank)

    def forward(self, tokens, query):
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        query = F.layer_norm(query, (query.shape[-1],))
        return [tuple((head[f"key_{boundary}"](tokens) * head[f"query_{boundary}"](query)).sum(-1)
                      * self.scale for boundary in ("start", "end", "presence")) for head in self.heads]


def span_loss(outputs, targets):
    loss = outputs[0][0].new_zeros(())
    for (start, end, presence), target in zip(outputs, targets):
        logit = torch.logsumexp(presence, dim=-1)
        loss = loss + F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(target is not None)))
        if target is not None:
            gold = torch.tensor(target, device=start.device)
            loss = loss + (F.cross_entropy(start[None], gold[:1])
                           + F.cross_entropy(end[None], gold[1:])) / 2
    return loss


def batch_outputs(encoder, heads, batch, device, pad_id):
    ids = torch.full((len(batch), max(len(e.input_ids) for e in batch)), pad_id,
                     dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, example in enumerate(batch):
        ids[i, :len(example.input_ids)] = torch.tensor(example.input_ids, device=device)
        mask[i, :len(example.input_ids)] = 1
    hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    return [heads(hidden[i, example.positions], hidden[i, 0]) for i, example in enumerate(batch)]


@torch.inference_mode()
def development_accuracy(encoder, heads, examples, device, pad_id, batch_size):
    encoder.eval()
    heads.eval()
    correct = 0
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        for outputs, example in zip(batch_outputs(encoder, heads, batch, device, pad_id), batch):
            for (start, end, presence), target in zip(outputs, example.targets):
                present = torch.logsumexp(presence, dim=-1).item() >= PRESENCE_THRESHOLD
                correct += int((target is None and not present) or
                               (target is not None and present and
                                (start.argmax().item(), end.argmax().item()) == target))
    return correct / (len(FIELDS) * len(examples))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="gt-csse/ML-cite-training")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model-id", default="google-bert/bert-base-cased")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/bert"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.grad_accum) < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch sizes and learning rate must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    snapshot = Path(snapshot_download(args.dataset_id, repo_type="dataset", revision=args.dataset_revision,
                                      allow_patterns=["data/train.jsonl", "data/dev.jsonl"]))
    config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
    if config.model_type != "bert":
        raise ValueError("this trainer requires a BERT model")
    model_revision = getattr(config, "_commit_hash", None) or args.model_revision
    tokenizer = BertTokenizerFast.from_pretrained(args.model_id, revision=model_revision)
    examples, exclusions, hashes = load_examples(tokenizer, snapshot, config.max_position_embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "exclusions.json", exclusions)
    encoder = BertModel.from_pretrained(args.model_id, revision=model_revision, config=config,
                                        add_pooling_layer=False).to(device)
    heads = SpanHeads(config.hidden_size).to(device)
    parameters = list(encoder.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset_revision": snapshot.name, "model_revision": model_revision,
                "dataset_sha256": hashes, "fields": list(FIELDS), "query_rank": 64,
                "presence_logit_threshold": PRESENCE_THRESHOLD, "max_span_tokens": MAX_SPAN_TOKENS,
                "rows": {split: len(rows) for split, rows in examples.items()},
                "device": str(device), "status": "running", "history": []}
    write_json(args.output_dir / "training.json", metadata)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        heads.train()
        order = list(examples["train"])
        random.shuffle(order)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(order), args.batch_size * args.grad_accum):
            group = order[offset:offset + args.batch_size * args.grad_accum]
            for micro in range(0, len(group), args.batch_size):
                batch = group[micro:micro + args.batch_size]
                outputs = batch_outputs(encoder, heads, batch, device, tokenizer.pad_token_id)
                loss = sum(span_loss(out, e.targets) for out, e in zip(outputs, batch))
                total_loss += loss.item()
                (loss / len(group)).backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        accuracy = development_accuracy(encoder, heads, examples["dev"], device,
                                        tokenizer.pad_token_id, args.batch_size)
        metadata["history"].append({"epoch": epoch, "train_loss": total_loss / len(order),
                                    "dev_accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            encoder.save_pretrained(args.output_dir / "bert")
            torch.save(heads.state_dict(), args.output_dir / "heads.pt")
            metadata.update(selected_epoch=epoch, selected_dev_accuracy=accuracy)
        write_json(args.output_dir / "training.json", metadata)
        print(f"epoch {epoch}: loss={total_loss / len(order):.4f} dev_accuracy={accuracy:.4%}", flush=True)
    metadata["status"] = "complete"
    write_json(args.output_dir / "training.json", metadata)
    print(f"Saved epoch {metadata['selected_epoch']} to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
