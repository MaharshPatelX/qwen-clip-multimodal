import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import json
import logging
from pathlib import Path


class BLEUScore:
    """BLEU score calculation for image captioning evaluation."""
    
    def __init__(self, max_order: int = 4, smooth: bool = False):
        self.max_order = max_order
        self.smooth = smooth
    
    def compute_bleu(self, reference_corpus: List[List[str]], translation_corpus: List[str]) -> Dict[str, float]:
        """
        Compute BLEU score for a corpus of translations.
        
        Args:
            reference_corpus: List of reference sentence lists (multiple refs per translation)
            translation_corpus: List of translation sentences
            
        Returns:
            Dictionary with BLEU scores
        """
        matches_by_order = [0] * self.max_order
        possible_matches_by_order = [0] * self.max_order
        reference_length = 0
        translation_length = 0
        
        for (references, translation) in zip(reference_corpus, translation_corpus):
            reference_length += min(len(r.split()) for r in references)
            translation_length += len(translation.split())
            
            merged_ref_ngram_counts = Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference.split(), self.max_order)
            
            translation_ngram_counts = self._get_ngrams(translation.split(), self.max_order)
            
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            
            for order in range(1, self.max_order + 1):
                possible_matches = len(translation.split()) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches
        
        precisions = [0] * self.max_order
        for i in range(0, self.max_order):
            if self.smooth:
                precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
                else:
                    precisions[i] = 0.0
        
        if min(precisions) > 0:
            p_log_sum = sum((1. / self.max_order) * np.log(p) for p in precisions)
            geo_mean = np.exp(p_log_sum)
        else:
            geo_mean = 0
        
        ratio = float(translation_length) / reference_length
        
        if ratio > 1.0:
            bp = 1.
        else:
            bp = np.exp(1 - 1. / ratio)
        
        bleu = geo_mean * bp
        
        return {
            'bleu': bleu,
            'bleu_1': precisions[0],
            'bleu_2': precisions[1],
            'bleu_3': precisions[2],
            'bleu_4': precisions[3],
            'brevity_penalty': bp,
            'length_ratio': ratio,
            'translation_length': translation_length,
            'reference_length': reference_length
        }
    
    def _get_ngrams(self, segment: List[str], max_order: int) -> Counter:
        """Extract n-grams up to max_order from segment."""
        ngram_counts = Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts


class ROUGEScore:
    """ROUGE score calculation for text summarization evaluation."""
    
    def __init__(self):
        pass
    
    def rouge_n(self, evaluated_sentences: List[str], reference_sentences: List[str], n: int = 1) -> float:
        """Calculate ROUGE-N score."""
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            return 0.0
        
        evaluated_ngrams = self._get_word_ngrams(n, evaluated_sentences)
        reference_ngrams = self._get_word_ngrams(n, reference_sentences)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)
        
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)
        
        if evaluated_count == 0:
            return 0.0
        
        if reference_count == 0:
            return 0.0
        
        return overlapping_count / reference_count
    
    def rouge_l(self, evaluated_sentences: List[str], reference_sentences: List[str]) -> float:
        """Calculate ROUGE-L score using Longest Common Subsequence."""
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            return 0.0
        
        evaluated_words = " ".join(evaluated_sentences).split()
        reference_words = " ".join(reference_sentences).split()
        
        lcs_length = self._lcs(evaluated_words, reference_words)
        
        if len(evaluated_words) == 0 or len(reference_words) == 0:
            return 0.0
        
        precision = lcs_length / len(evaluated_words)
        recall = lcs_length / len(reference_words)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _get_word_ngrams(self, n: int, sentences: List[str]) -> set:
        """Generate n-grams from sentences."""
        ngrams = set()
        for sentence in sentences:
            words = sentence.lower().split()
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                ngrams.add(ngram)
        return ngrams
    
    def _lcs(self, x: List[str], y: List[str]) -> int:
        """Calculate Longest Common Subsequence length."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]


class CIDErScore:
    """CIDEr score calculation for image captioning."""
    
    def __init__(self, n: int = 4, sigma: float = 6.0):
        self.n = n
        self.sigma = sigma
    
    def compute_cider(self, gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> float:
        """
        Compute CIDEr score.
        
        Args:
            gts: Dictionary of ground truth captions {image_id: [captions]}
            res: Dictionary of generated captions {image_id: [caption]}
            
        Returns:
            CIDEr score
        """
        # Compute document frequencies
        document_frequency = Counter()
        
        for img_id in gts.keys():
            # Ground truth captions
            for caption in gts[img_id]:
                ngrams = self._get_ngrams(caption, self.n)
                for ngram in ngrams:
                    document_frequency[ngram] += 1
        
        # Compute CIDEr for each image
        cider_scores = []
        
        for img_id in gts.keys():
            # Get n-grams for ground truth and generated captions
            gt_ngrams = []
            for caption in gts[img_id]:
                gt_ngrams.extend(self._get_ngrams(caption, self.n))
            
            res_ngrams = []
            for caption in res[img_id]:
                res_ngrams.extend(self._get_ngrams(caption, self.n))
            
            # Compute TF-IDF vectors
            gt_vec = self._compute_tf_idf(gt_ngrams, document_frequency, len(gts))
            res_vec = self._compute_tf_idf(res_ngrams, document_frequency, len(gts))
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(gt_vec, res_vec)
            cider_scores.append(similarity)
        
        return np.mean(cider_scores)
    
    def _get_ngrams(self, caption: str, n: int) -> List[tuple]:
        """Extract n-grams from caption."""
        words = caption.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i + n]))
        return ngrams
    
    def _compute_tf_idf(self, ngrams: List[tuple], doc_freq: Counter, num_docs: int) -> Dict[tuple, float]:
        """Compute TF-IDF vector for n-grams."""
        tf = Counter(ngrams)
        tf_idf = {}
        
        for ngram, count in tf.items():
            tf_score = count / len(ngrams) if len(ngrams) > 0 else 0
            idf_score = np.log(num_docs / (doc_freq[ngram] + 1))
            tf_idf[ngram] = tf_score * idf_score
        
        return tf_idf
    
    def _cosine_similarity(self, vec1: Dict[tuple, float], vec2: Dict[tuple, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        all_ngrams = set(vec1.keys()) | set(vec2.keys())
        
        for ngram in all_ngrams:
            v1 = vec1.get(ngram, 0)
            v2 = vec2.get(ngram, 0)
            
            dot_product += v1 * v2
            norm1 += v1 ** 2
            norm2 += v2 ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))


class VQAAccuracy:
    """VQA accuracy calculation."""
    
    def __init__(self):
        pass
    
    def compute_accuracy(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute VQA accuracy.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answer lists
            
        Returns:
            Dictionary with accuracy metrics
        """
        assert len(predictions) == len(references)
        
        exact_matches = 0
        normalized_matches = 0
        
        for pred, refs in zip(predictions, references):
            # Exact match
            if pred in refs:
                exact_matches += 1
            
            # Normalized match
            pred_normalized = self._normalize_answer(pred)
            refs_normalized = [self._normalize_answer(ref) for ref in refs]
            
            if pred_normalized in refs_normalized:
                normalized_matches += 1
        
        total = len(predictions)
        
        return {
            'exact_accuracy': exact_matches / total if total > 0 else 0.0,
            'normalized_accuracy': normalized_matches / total if total > 0 else 0.0,
            'total_samples': total
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer.strip()


class MultimodalEvaluator:
    """Comprehensive evaluator for multimodal models."""
    
    def __init__(self):
        self.bleu_scorer = BLEUScore()
        self.rouge_scorer = ROUGEScore()
        self.cider_scorer = CIDErScore()
        self.vqa_scorer = VQAAccuracy()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_captioning(
        self,
        predictions: List[str],
        references: List[List[str]],
        image_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate image captioning performance.
        
        Args:
            predictions: List of generated captions
            references: List of reference caption lists
            image_ids: Optional list of image IDs
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {len(predictions)} captions...")
        
        # BLEU scores
        bleu_scores = self.bleu_scorer.compute_bleu(references, predictions)
        
        # ROUGE scores
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, refs in zip(predictions, references):
            rouge_1 = self.rouge_scorer.rouge_n([pred], refs, n=1)
            rouge_2 = self.rouge_scorer.rouge_n([pred], refs, n=2)
            rouge_l = self.rouge_scorer.rouge_l([pred], refs)
            
            rouge_1_scores.append(rouge_1)
            rouge_2_scores.append(rouge_2)
            rouge_l_scores.append(rouge_l)
        
        # CIDEr score
        if image_ids is None:
            image_ids = [f"img_{i}" for i in range(len(predictions))]
        
        gts = {img_id: refs for img_id, refs in zip(image_ids, references)}
        res = {img_id: [pred] for img_id, pred in zip(image_ids, predictions)}
        
        cider_score = self.cider_scorer.compute_cider(gts, res)
        
        results = {
            'bleu': bleu_scores,
            'rouge_1': np.mean(rouge_1_scores),
            'rouge_2': np.mean(rouge_2_scores),
            'rouge_l': np.mean(rouge_l_scores),
            'cider': cider_score,
            'num_samples': len(predictions)
        }
        
        self.logger.info(f"Captioning evaluation complete:")
        self.logger.info(f"  BLEU-4: {bleu_scores['bleu_4']:.4f}")
        self.logger.info(f"  ROUGE-L: {results['rouge_l']:.4f}")
        self.logger.info(f"  CIDEr: {results['cider']:.4f}")
        
        return results
    
    def evaluate_vqa(
        self,
        predictions: List[str],
        references: List[List[str]],
        questions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate VQA performance.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answer lists
            questions: Optional list of questions
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {len(predictions)} VQA predictions...")
        
        accuracy_scores = self.vqa_scorer.compute_accuracy(predictions, references)
        
        # Additional analysis by question type if questions provided
        question_type_accuracy = {}
        if questions is not None:
            question_types = self._categorize_questions(questions)
            
            for qtype in set(question_types):
                qtype_preds = [pred for pred, qt in zip(predictions, question_types) if qt == qtype]
                qtype_refs = [ref for ref, qt in zip(references, question_types) if qt == qtype]
                
                if qtype_preds:
                    qtype_acc = self.vqa_scorer.compute_accuracy(qtype_preds, qtype_refs)
                    question_type_accuracy[qtype] = qtype_acc
        
        results = {
            **accuracy_scores,
            'question_type_accuracy': question_type_accuracy
        }
        
        self.logger.info(f"VQA evaluation complete:")
        self.logger.info(f"  Exact Accuracy: {accuracy_scores['exact_accuracy']:.4f}")
        self.logger.info(f"  Normalized Accuracy: {accuracy_scores['normalized_accuracy']:.4f}")
        
        return results
    
    def _categorize_questions(self, questions: List[str]) -> List[str]:
        """Categorize questions by type."""
        categories = []
        
        for question in questions:
            q_lower = question.lower()
            
            if q_lower.startswith(('what', 'which')):
                categories.append('what')
            elif q_lower.startswith('where'):
                categories.append('where')
            elif q_lower.startswith('when'):
                categories.append('when')
            elif q_lower.startswith('who'):
                categories.append('who')
            elif q_lower.startswith('why'):
                categories.append('why')
            elif q_lower.startswith('how'):
                categories.append('how')
            elif q_lower.startswith(('is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would')):
                categories.append('yes_no')
            else:
                categories.append('other')
        
        return categories
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Test evaluation metrics
    evaluator = MultimodalEvaluator()
    
    # Test captioning evaluation
    predictions = ["A cat sitting on a table", "A dog running in the park"]
    references = [
        ["A cat is sitting on the table", "Cat on table"], 
        ["Dog running in park", "A dog is running"]
    ]
    
    caption_results = evaluator.evaluate_captioning(predictions, references)
    print("Captioning results:", caption_results)
    
    # Test VQA evaluation  
    vqa_predictions = ["cat", "blue"]
    vqa_references = [["cat", "kitten"], ["blue", "navy"]]
    vqa_questions = ["What animal is this?", "What color is the sky?"]
    
    vqa_results = evaluator.evaluate_vqa(vqa_predictions, vqa_references, vqa_questions)
    print("VQA results:", vqa_results)
    
    print("Evaluation metrics working correctly!")