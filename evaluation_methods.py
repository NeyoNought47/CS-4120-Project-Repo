import evaluate
from typing import List, Dict, Any, Optional, Union, Tuple


def compute_metrics(
    sources: List[str],
    references: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Generic text-level metrics computation (BLEU, chrF, METEOR, COMET).

    This is the original function used by the RNN pipeline and can also be
    used directly for MT5 if you already have decoded text predictions.
    """
    results: Dict[str, float] = {}

    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")
    metric_meteor = evaluate.load("meteor")
    metric_comet = evaluate.load("comet")

    # sacrebleu / chrf expect list[list[str]] for references
    formatted_refs = [[r] for r in references]

    bleu_res = metric_bleu.compute(predictions=predictions, references=formatted_refs)
    results["BLEU"] = bleu_res["score"]
    print(f"BLEU: {results['BLEU']:.2f}")

    chrf_res = metric_chrf.compute(predictions=predictions, references=formatted_refs)
    results["chrF"] = chrf_res["score"]
    print(f"chrF: {results['chrF']:.2f}")

    meteor_res = metric_meteor.compute(predictions=predictions, references=references)
    results["METEOR"] = meteor_res["meteor"]
    print(f"METEOR: {results['METEOR']:.4f}")

    comet_res = metric_comet.compute(
        predictions=predictions, references=references, sources=sources
    )
    results["COMET"] = comet_res["mean_score"]
    print(f"COMET: {results['COMET']:.4f}")

    return results


def compute_metrics_mt5_from_ids(
    eval_predictions: Union[
        Any,  # e.g. transformers.EvalPrediction
        Tuple[Any, Any],  # (predictions, label_ids)
    ],
    tokenizer,
    sources: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper for MT5 / Hugging Face Trainer-style outputs.

    Expects either:
      - An EvalPrediction-like object with `.predictions` and `.label_ids`, or
      - A tuple (predictions, label_ids).

    Both `predictions` and `label_ids` are arrays of token ids. This function
    decodes them to text and then delegates to `compute_metrics`.
    """
    # Unpack predictions / labels from different possible formats
    if isinstance(eval_predictions, tuple):
        preds, labels = eval_predictions
    else:
        preds = getattr(eval_predictions, "predictions")
        labels = getattr(eval_predictions, "label_ids")

    # Replace ignored index (-100) in labels before decoding
    import numpy as np

    if isinstance(labels, list):
        labels = np.array(labels)

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # Decode to strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    if sources is None:
        # If sources are not provided, just use empty strings of the same length
        sources = [""] * len(decoded_preds)

    return compute_metrics(sources=sources, references=decoded_labels, predictions=decoded_preds)