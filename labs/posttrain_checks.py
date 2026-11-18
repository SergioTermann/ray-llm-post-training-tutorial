"""Dependency-free correctness labs; synthetic inputs are not model evaluations."""
import hashlib
import json
import math
import random
import re
import unicodedata


def canonical(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def audit_splits(splits):
    """Find exact normalized prompt leakage; not semantic or benchmark decontamination."""
    seen_ids, seen_prompts, issues = {}, {}, []
    for split, rows in splits.items():
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip()
                       for k in ("id", "prompt", "answer")):
                issues.append({"split": split, "kind": "invalid_schema"})
                continue
            key = hashlib.sha256(canonical(row["prompt"]).encode()).hexdigest()
            if row["id"] in seen_ids:
                issues.append({"split": split, "kind": "duplicate_id", "id": row["id"]})
            if key in seen_prompts and seen_prompts[key] != split:
                issues.append({"split": split, "kind": "cross_split_prompt", "id": row["id"]})
            seen_ids[row["id"]] = split
            seen_prompts.setdefault(key, split)
    return issues


def assistant_labels(input_ids, attention_mask, assistant_mask):
    """Masks come from a validated tokenizer template, not string offsets."""
    if not (len(input_ids) == len(attention_mask) == len(assistant_mask)):
        raise ValueError("Mask lengths differ")
    if any(x not in (0, 1) for x in attention_mask + assistant_mask):
        raise ValueError("Masks must be binary")
    labels = [token if attn and target else -100
              for token, attn, target in zip(input_ids, attention_mask, assistant_mask)]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No supervised next-token targets remain after shifting")
    return labels


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Sequence log probabilities are sums over completion tokens only."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return softplus(-beta * margin)


def toy_dpo_update(steps=30, learning_rate=0.5, beta=0.2):
    """Two-answer policy: log odds theta, uniform fixed reference, chosen always wins."""
    theta = 0.0
    trace = []
    for step in range(steps + 1):
        chosen_logp, rejected_logp = -softplus(-theta), -softplus(theta)
        loss = dpo_loss(chosen_logp, rejected_logp, -math.log(2), -math.log(2), beta)
        trace.append({"step": step, "loss": loss, "chosen_probability": math.exp(chosen_logp)})
        if step != steps:
            theta += learning_rate * beta * math.exp(-softplus(beta * theta))
    return trace


def paired_bootstrap(base, candidate, repeats=2000, seed=42):
    """Resample independent prompt-level paired scores; interval is illustrative percentile CI."""
    if not base or len(base) != len(candidate):
        raise ValueError("Need equally sized non-empty paired scores")
    delta = [c - b for b, c in zip(base, candidate)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(repeats))
    return {"mean_delta": sum(delta) / len(delta),
            "ci95": [means[int(0.025 * (repeats - 1))], means[int(0.975 * (repeats - 1))]]}


def self_check():
    # The same ID used for EOS and PAD must be supervised only at the real EOS position.
    labels = assistant_labels([10, 11, 20, 2, 2], [1, 1, 1, 1, 0], [0, 0, 1, 1, 0])
    assert labels == [-100, -100, 20, 2, -100]
    assert len(labels[1:]) == 4
    try:
        assistant_labels([10, 11], [1, 1], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty supervision was accepted")
    leaked = {"train": [{"id": "a", "prompt": "What is 2+2?", "answer": "4"}],
              "test": [{"id": "b", "prompt": " what IS 2+2? ", "answer": "4"}]}
    assert audit_splits(leaked)[0]["kind"] == "cross_split_prompt"
    beta, theta, eps = 0.2, 0.4, 1e-5
    finite_difference = (softplus(-beta*(theta+eps)) - softplus(-beta*(theta-eps))) / (2*eps)
    analytic = -beta * math.exp(-softplus(beta*theta))
    assert math.isclose(finite_difference, analytic, rel_tol=1e-6)
    assert math.isclose(dpo_loss(-2, -3, -2, -3), math.log(2))
    assert math.isfinite(dpo_loss(-1e4, 0, 0, 0))
    trace = toy_dpo_update()
    assert trace[-1]["loss"] < trace[0]["loss"]
    assert trace[-1]["chosen_probability"] > trace[0]["chosen_probability"]
    assert paired_bootstrap([0, 1], [0, 1])["ci95"] == [0, 0]
    return {"checks": "passed", "toy_initial": trace[0], "toy_final": trace[-1]}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2))
