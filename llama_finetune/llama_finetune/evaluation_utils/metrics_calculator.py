from typing import List, Dict, Tuple
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu

class MetricsCalculator:
    @staticmethod
    def compute_bleu(reference: str, candidate: str) -> float:
        """Compute BLEU score between reference and candidate code."""
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        smoothie = SmoothingFunction().method4
        return sentence_bleu(
            [reference_tokens], candidate_tokens, smoothing_function=smoothie
        )
        
    @staticmethod
    def compute_chrf(reference: str, candidate: str) -> float:
        """Compute ChrF score between reference and candidate code."""
        chrf = sacrebleu.sentence_chrf(
            hypothesis=candidate,
            references=[reference],
            char_order=6,  # Default n-gram order for characters
            word_order=0,  # No word n-grams, only character n-grams
            beta=2.0       # More weight to recall than precision
        )
        return chrf.score / 100.0  # Normalize to 0-1 range

    @staticmethod
    def calculate_metrics(
        references: List[str], generated_codes: List[str]
    ) -> Tuple[List[Dict], float]:
        """Calculate BLEU scores for each pair and average."""
        results = []
        bleu_scores = []

        for ref, gen in zip(references, generated_codes):
            bleu = MetricsCalculator.compute_bleu(ref, gen)
            bleu_scores.append(bleu)
            results.append({"reference": ref, "generated": gen, "bleu": bleu})

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        return results, avg_bleu