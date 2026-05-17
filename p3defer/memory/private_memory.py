"""
Private Memory Module for P3Defer.

Implements a dynamic growing list of private tokens (Section 2.4 of the paper).
Uses Levenshtein distance to detect private tokens in queries and masks them
with semantically similar but non-identifying alternatives.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set

import Levenshtein

logger = logging.getLogger(__name__)


# Common private token categories and their generic replacements
_DEFAULT_REPLACEMENTS = {
    "person": ["someone", "a person", "an individual", "a student", "a worker"],
    "location": ["a place", "somewhere", "a city", "a town", "a location"],
    "organization": ["a company", "an organization", "a group", "an institution"],
    "email": ["an email address"],
    "phone": ["a phone number"],
    "date_of_birth": ["a date"],
    "ssn": ["an ID number"],
    "medical": ["a medical condition", "a health issue"],
}


class PrivateMemory:
    """Dynamic growing list of private tokens with masking capability.

    The private memory pre-stores known private tokens from a corpus and
    dynamically grows as new private tokens are encountered. When a
    privacy-sensitive query is detected, the memory identifies private
    tokens via Levenshtein distance and masks them with similar
    non-identifying alternatives.

    Attributes:
        tokens: Dictionary mapping private tokens to their categories.
        threshold: Levenshtein distance threshold for fuzzy matching.
        replacements: Category-to-replacement mapping for masking.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        seed_tokens: Optional[Dict[str, str]] = None,
        replacements: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the private memory.

        Args:
            threshold: Normalized Levenshtein distance threshold (0-1).
                Tokens with distance below this threshold are considered matches.
            seed_tokens: Initial dictionary of {token: category} pairs.
            replacements: Custom category-to-replacement mapping.
        """
        self.threshold = threshold
        self.tokens: Dict[str, str] = seed_tokens or {}
        self.replacements = replacements or _DEFAULT_REPLACEMENTS
        self._replacement_idx: Dict[str, int] = {}
        self._detected_history: List[Dict] = []

    @property
    def size(self) -> int:
        """Number of tokens in the private memory."""
        return len(self.tokens)

    def add_token(self, token: str, category: str = "person") -> None:
        """Add a private token to the memory.

        Args:
            token: The private token string (e.g., a person's name).
            category: The category of the token (e.g., 'person', 'location').
        """
        normalized = token.strip().lower()
        if normalized and normalized not in self.tokens:
            self.tokens[normalized] = category
            logger.debug("Added private token '%s' (category: %s)", token, category)

    def add_tokens_from_corpus(self, texts: List[str], labels: List[int]) -> int:
        """Extract and store private tokens from a labeled corpus.

        Scans texts that are labeled as containing privacy-sensitive content
        and extracts likely private tokens (proper nouns, names, etc.).

        Args:
            texts: List of text strings.
            labels: Binary labels (1 = contains private info, 0 = does not).

        Returns:
            Number of new tokens added.
        """
        added = 0
        for text, label in zip(texts, labels):
            if label == 1:
                names = self._extract_proper_nouns(text)
                for name in names:
                    if name.lower() not in self.tokens:
                        self.add_token(name, "person")
                        added += 1
        logger.info("Added %d private tokens from corpus of %d texts", added, len(texts))
        return added

    def _extract_proper_nouns(self, text: str) -> List[str]:
        """Extract likely proper nouns (names) from text using heuristics.

        Uses capitalization patterns and common name indicators to identify
        potential private tokens without requiring NER models.

        Args:
            text: Input text string.

        Returns:
            List of extracted proper noun strings.
        """
        names = set()
        # Split into sentences, then look for capitalized words not at sentence start
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                clean = re.sub(r'[^a-zA-Z\'-]', '', word)
                if not clean:
                    continue
                # Skip first word of sentence (always capitalized)
                if i == 0:
                    # But if it looks like a name (not a common word), keep it
                    if clean[0].isupper() and len(clean) > 1 and clean.lower() not in _COMMON_WORDS:
                        names.add(clean)
                    continue
                # Capitalized word mid-sentence is likely a proper noun
                if clean[0].isupper() and len(clean) > 1 and clean.lower() not in _COMMON_WORDS:
                    names.add(clean)
        return list(names)

    def detect_private_tokens(self, text: str) -> List[Tuple[str, str, str]]:
        """Detect private tokens in a text using Levenshtein distance matching.

        For each word in the text, computes the normalized Levenshtein distance
        against all tokens in the memory. Words with distance below the threshold
        are flagged as private.

        Args:
            text: Input text to scan.

        Returns:
            List of (matched_word, memory_token, category) tuples.
        """
        matches = []
        words = text.split()
        seen = set()

        for word in words:
            clean = re.sub(r'[^a-zA-Z\'-]', '', word).lower()
            if not clean or len(clean) < 2 or clean in seen:
                continue
            seen.add(clean)

            for token, category in self.tokens.items():
                if len(token) < 2:
                    continue
                # Normalized Levenshtein distance
                max_len = max(len(clean), len(token))
                dist = Levenshtein.distance(clean, token) / max_len
                if dist <= self.threshold:
                    matches.append((word, token, category))
                    break  # One match per word is sufficient

        return matches

    def mask_query(self, text: str) -> Tuple[str, List[Dict], int]:
        """Mask private tokens in a query with generic replacements.

        Identifies private tokens in the text and replaces them with
        category-appropriate generic alternatives while preserving
        the original query intent.

        Args:
            text: Input query text.

        Returns:
            Tuple of (masked_text, list_of_replacements_made, num_tokens_masked).
        """
        detections = self.detect_private_tokens(text)
        if not detections:
            return text, [], 0

        masked_text = text
        replacements_made = []

        for original_word, memory_token, category in detections:
            replacement = self._get_replacement(category)
            # Replace the original word (case-insensitive, whole word)
            pattern = re.compile(re.escape(original_word), re.IGNORECASE)
            masked_text = pattern.sub(replacement, masked_text, count=1)
            replacements_made.append({
                "original": original_word,
                "matched_token": memory_token,
                "category": category,
                "replacement": replacement,
            })

        # Record detection for analysis
        self._detected_history.append({
            "original_text": text[:100],
            "num_masked": len(detections),
            "detections": replacements_made,
        })

        return masked_text, replacements_made, len(detections)

    def _get_replacement(self, category: str) -> str:
        """Get a replacement string for a given category.

        Cycles through available replacements to add variety.

        Args:
            category: Token category (e.g., 'person').

        Returns:
            Replacement string.
        """
        options = self.replacements.get(category, ["[REDACTED]"])
        idx = self._replacement_idx.get(category, 0)
        replacement = options[idx % len(options)]
        self._replacement_idx[category] = idx + 1
        return replacement

    def compute_leakage_rate(
        self, original_texts: List[str], masked_texts: List[str]
    ) -> Dict[str, float]:
        """Compute the privacy leakage rate r(leakage).

        Measures the ratio of private tokens that remain unmasked after
        applying the private memory masking.

        Args:
            original_texts: List of original query texts.
            masked_texts: List of masked query texts.

        Returns:
            Dictionary with leakage statistics.
        """
        total_private = 0
        leaked_private = 0

        for orig, masked in zip(original_texts, masked_texts):
            detections = self.detect_private_tokens(orig)
            total_private += len(detections)
            # Check if any detected tokens still appear in masked text
            for original_word, _, _ in detections:
                if original_word.lower() in masked.lower():
                    leaked_private += 1

        leakage_rate = leaked_private / max(total_private, 1)
        return {
            "total_private_tokens": total_private,
            "leaked_tokens": leaked_private,
            "leakage_rate": leakage_rate * 100,  # percentage
        }

    def compute_detection_metrics(
        self, texts: List[str], labels: List[int]
    ) -> Dict[str, float]:
        """Compute precision and recall for privacy-sensitive query identification.

        Args:
            texts: List of query texts.
            labels: Binary labels (1 = contains private info).

        Returns:
            Dictionary with precision, recall, and F1 scores.
        """
        tp = fp = fn = tn = 0
        for text, label in zip(texts, labels):
            detections = self.detect_private_tokens(text)
            predicted = 1 if len(detections) > 0 else 0
            if predicted == 1 and label == 1:
                tp += 1
            elif predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def save(self, path: str) -> None:
        """Save the private memory to a JSON file."""
        import json
        data = {
            "tokens": self.tokens,
            "threshold": self.threshold,
            "history_size": len(self._detected_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved private memory (%d tokens) to %s", self.size, path)

    def load(self, path: str) -> None:
        """Load the private memory from a JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.tokens = data["tokens"]
        self.threshold = data.get("threshold", self.threshold)
        logger.info("Loaded private memory (%d tokens) from %s", self.size, path)


# Common English words to exclude from proper noun detection
_COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "although", "since",
    "until", "unless", "that", "which", "who", "whom", "this", "these",
    "those", "what", "every", "many", "much", "any", "also", "about",
    "up", "down", "now", "new", "old", "first", "last", "long", "great",
    "little", "right", "big", "high", "small", "large", "next", "early",
    "young", "important", "public", "bad", "good", "best", "well", "way",
    "day", "time", "year", "people", "work", "number", "part", "place",
    "case", "week", "company", "system", "program", "question", "answer",
    "home", "hand", "world", "life", "school", "state", "family", "student",
    "group", "country", "problem", "fact", "however", "therefore", "thus",
    "hence", "moreover", "furthermore", "additionally", "meanwhile",
    "assume", "given", "please", "output", "solve", "check", "contains",
    "personal", "information", "let", "think", "step", "example", "examples",
    "total", "many", "much", "each", "every", "purchased", "gave", "twice",
    "half", "times", "less", "remaining", "average", "higher", "lower",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
