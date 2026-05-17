"""
Evaluation Module for P3Defer.

Implements all evaluation metrics from the paper (Table 1):
- Accuracy (Acc.): Exact match for QA tasks
- ROUGE-1, ROUGE-L: For summarization tasks
- Coverage Rate (CR): Fraction of queries answered (not abstained)
- Server Coverage Rate (SCR): Fraction of queries deferred to server
- Leakage Rate r(leakage): Fraction of private tokens leaked to server
- Privacy Precision/Recall: Detection accuracy for private queries
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores.

    Args:
        predictions: List of predicted texts.
        references: List of reference texts.

    Returns:
        Dictionary with rouge1, rouge2, rougeL scores.
    """
    try:
        from rouge_score import rouge_scorer, scoring
    except ImportError:
        logger.warning("rouge_score not installed. Using simple overlap metric.")
        return _simple_rouge(predictions, references)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        score = scorer.score(ref, pred)
        aggregator.add_scores(score)

    result = aggregator.aggregate()
    return {
        "rouge1": result["rouge1"].mid.fmeasure * 100,
        "rouge2": result["rouge2"].mid.fmeasure * 100,
        "rougeL": result["rougeL"].mid.fmeasure * 100,
    }


def _simple_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Simple word-overlap based ROUGE approximation."""
    scores = {"rouge1": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            continue
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / max(len(pred_tokens), 1)
        recall = overlap / max(len(ref_tokens), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        scores["rouge1"].append(f1 * 100)
        scores["rougeL"].append(f1 * 100)  # Simplified
    return {k: np.mean(v) if v else 0.0 for k, v in scores.items()}


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute exact match accuracy for QA tasks.

    Extracts numerical answers and compares them.

    Args:
        predictions: List of predicted answer texts.
        references: List of reference answer texts.

    Returns:
        Accuracy as a percentage.
    """
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        pred_num = _extract_number(pred)
        ref_num = _extract_number(ref)
        if ref_num is not None:
            total += 1
            if pred_num is not None and abs(pred_num - ref_num) < 1e-5:
                correct += 1
        else:
            total += 1
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
    return (correct / max(total, 1)) * 100


def _extract_number(text: str) -> Optional[float]:
    """Extract the final number from a text string."""
    text = str(text).replace(",", "").strip()
    # Look for #### pattern first (GSM8K format)
    match = re.search(r'####\s*([\d.-]+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Find all numbers and return the last one
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


class CascadeEvaluator:
    """Full evaluation pipeline for the P3Defer cascade system.

    Evaluates the cascade on all metrics from the paper:
    quality metrics (Acc, ROUGE), efficiency metrics (CR, SCR),
    and privacy metrics (leakage rate, precision, recall).
    """

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        output_dir: str = "./output/eval_results",
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        actions: List[int],
        privacy_labels: List[int],
        privacy_predictions: List[int],
        original_queries: Optional[List[str]] = None,
        masked_queries: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run full evaluation.

        Args:
            predictions: List of generated answer texts.
            references: List of reference answer texts.
            actions: List of deferral actions (0=local, 1=defer, 2=abstain).
            privacy_labels: List of ground-truth privacy labels (0/1).
            privacy_predictions: List of predicted privacy labels (0/1).
            original_queries: Optional list of original query texts.
            masked_queries: Optional list of masked query texts (for leakage).

        Returns:
            Dictionary with all evaluation metrics.
        """
        results = {}
        n = len(predictions)

        # Quality metrics
        if self.dataset_name == "gsm8k":
            results["accuracy"] = compute_accuracy(predictions, references)
        rouge = compute_rouge(predictions, references)
        results.update(rouge)

        # Coverage Rate (CR): fraction of queries answered (not abstained)
        answered = sum(1 for a in actions if a != 2)
        results["coverage_rate"] = (answered / max(n, 1)) * 100

        # Server Coverage Rate (SCR): fraction deferred to server
        deferred = sum(1 for a in actions if a == 1)
        results["server_coverage_rate"] = (deferred / max(n, 1)) * 100

        # Local answer rate
        local = sum(1 for a in actions if a == 0)
        results["local_rate"] = (local / max(n, 1)) * 100

        # Abstain rate
        abstained = sum(1 for a in actions if a == 2)
        results["abstain_rate"] = (abstained / max(n, 1)) * 100

        # Privacy detection metrics
        tp = sum(1 for p, l in zip(privacy_predictions, privacy_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(privacy_predictions, privacy_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(privacy_predictions, privacy_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(privacy_predictions, privacy_labels) if p == 0 and l == 0)

        results["privacy_precision"] = (tp / max(tp + fp, 1)) * 100
        results["privacy_recall"] = (tp / max(tp + fn, 1)) * 100
        results["privacy_f1"] = (
            2 * results["privacy_precision"] * results["privacy_recall"]
            / max(results["privacy_precision"] + results["privacy_recall"], 1e-8)
        )
        results["privacy_accuracy"] = ((tp + tn) / max(n, 1)) * 100

        # Leakage rate
        if original_queries and masked_queries:
            results["leakage_rate"] = self._compute_leakage(
                original_queries, masked_queries, actions, privacy_labels
            )
        else:
            # Estimate leakage: private queries deferred without masking
            private_deferred = sum(
                1 for a, l in zip(actions, privacy_labels) if a == 1 and l == 1
            )
            total_private = sum(privacy_labels)
            results["leakage_rate"] = (private_deferred / max(total_private, 1)) * 100

        # Quality by action type
        local_preds = [p for p, a in zip(predictions, actions) if a == 0]
        local_refs = [r for r, a in zip(references, actions) if a == 0]
        defer_preds = [p for p, a in zip(predictions, actions) if a == 1]
        defer_refs = [r for r, a in zip(references, actions) if a == 1]

        if local_preds:
            if self.dataset_name == "gsm8k":
                results["local_accuracy"] = compute_accuracy(local_preds, local_refs)
            local_rouge = compute_rouge(local_preds, local_refs)
            results["local_rougeL"] = local_rouge.get("rougeL", 0.0)

        if defer_preds:
            if self.dataset_name == "gsm8k":
                results["defer_accuracy"] = compute_accuracy(defer_preds, defer_refs)
            defer_rouge = compute_rouge(defer_preds, defer_refs)
            results["defer_rougeL"] = defer_rouge.get("rougeL", 0.0)

        # Save results
        results_path = os.path.join(self.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to %s", results_path)

        return results

    def _compute_leakage(
        self,
        original_queries: List[str],
        masked_queries: List[str],
        actions: List[int],
        privacy_labels: List[int],
    ) -> float:
        """Compute privacy leakage rate for deferred queries."""
        total_private_tokens = 0
        leaked_tokens = 0

        for orig, masked, action, label in zip(
            original_queries, masked_queries, actions, privacy_labels
        ):
            if action != 1 or label != 1:
                continue

            # Find words in original that are proper nouns (potential private tokens)
            orig_words = orig.split()
            for word in orig_words:
                clean = re.sub(r'[^a-zA-Z\'-]', '', word)
                if clean and clean[0].isupper() and len(clean) > 1:
                    total_private_tokens += 1
                    if clean.lower() in masked.lower():
                        leaked_tokens += 1

        return (leaked_tokens / max(total_private_tokens, 1)) * 100

    def print_results(self, results: Dict[str, float]) -> str:
        """Format results as a readable table."""
        lines = [
            "=" * 60,
            f"P3Defer Evaluation Results ({self.dataset_name.upper()})",
            "=" * 60,
            "",
            "Quality Metrics:",
        ]
        if "accuracy" in results:
            lines.append(f"  Accuracy:              {results['accuracy']:.2f}%")
        lines.extend([
            f"  ROUGE-1:               {results.get('rouge1', 0):.2f}",
            f"  ROUGE-L:               {results.get('rougeL', 0):.2f}",
            "",
            "Efficiency Metrics:",
            f"  Coverage Rate (CR):    {results.get('coverage_rate', 0):.2f}%",
            f"  Server Coverage (SCR): {results.get('server_coverage_rate', 0):.2f}%",
            f"  Local Rate:            {results.get('local_rate', 0):.2f}%",
            f"  Abstain Rate:          {results.get('abstain_rate', 0):.2f}%",
            "",
            "Privacy Metrics:",
            f"  Privacy Precision:     {results.get('privacy_precision', 0):.2f}%",
            f"  Privacy Recall:        {results.get('privacy_recall', 0):.2f}%",
            f"  Privacy F1:            {results.get('privacy_f1', 0):.2f}%",
            f"  Leakage Rate:          {results.get('leakage_rate', 0):.2f}%",
            "",
            "=" * 60,
        ])
        output = "\n".join(lines)
        print(output)
        return output
