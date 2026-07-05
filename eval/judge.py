"""Shared accuracy evaluation — LLM-as-a-Judge (HF InferenceClient) + BERTScore
(evaluate.load, baseline-rescaled). Single implementation used by eval/evaluate.py
(offline benchmark) and api/app.py (live dashboard) so the two never drift apart.
"""
import os
import warnings

from dotenv import load_dotenv
load_dotenv()

HF_JUDGE_MODEL = os.getenv('HF_JUDGE_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')

_judge_client = None
_bertscore_metric = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        from huggingface_hub import InferenceClient
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise RuntimeError("HF_TOKEN environment variable is not set")
        _judge_client = InferenceClient(model=HF_JUDGE_MODEL, token=token)
    return _judge_client


JUDGE_PROMPT = """Grade the system's answer.
Question: {q}
Correct answer: {correct}
System answer: {answer}

Reply with only PASS or FAIL.
PASS = the system answer correctly addresses the question with no major errors.
FAIL = the answer is wrong, missing, or contradicts the correct answer."""

_GEMINI_FALLBACK_JUDGE_PROMPT = """You are an evaluator. Respond with exactly one word: PASS or FAIL.
PASS if the prediction contains the key facts from the ground truth and addresses the question, even if worded differently or with additional context.
FAIL only if the prediction is clearly wrong, contradicts the ground truth, or is completely irrelevant.

Question: {q}
Ground Truth: {correct}
Prediction: {answer}

Answer (PASS or FAIL):"""


def _is_hf_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return '402' in msg or '429' in msg or 'quota' in msg or 'credit' in msg


def llm_judge(question: str, ground_truth: str, prediction: str) -> str:
    verdict, _ = llm_judge_with_source(question, ground_truth, prediction)
    return verdict


def llm_judge_with_source(question: str, ground_truth: str, prediction: str) -> tuple[str, str]:
    """Same grading as llm_judge, but also returns which judge actually produced the
    verdict ('huggingface' or 'gemini_fallback') for transparent reporting -- per spec,
    HF's InferenceClient is the primary judge; Gemini is only used when HF genuinely
    can't be reached (exhausted monthly credits, rate limit), never silently preferred."""
    try:
        client = _get_judge_client()
        prompt = JUDGE_PROMPT.format(q=question, correct=ground_truth, answer=prediction)
        response = client.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.upper()
        return ('PASS' if 'PASS' in verdict else 'FAIL'), 'huggingface'
    except Exception as e:
        if not _is_hf_quota_error(e):
            raise
        from pipelines.utils import setup_gemini, gemini_generate
        client = setup_gemini()
        prompt = _GEMINI_FALLBACK_JUDGE_PROMPT.format(q=question, correct=ground_truth, answer=prediction)
        response = gemini_generate(client, prompt, max_tokens=5)
        return ('PASS' if 'PASS' in response.upper() else 'FAIL'), 'gemini_fallback'


def compute_bertscore(predictions: list, references: list) -> dict:
    """Real, per-hackathon-spec BERTScore: evaluate.load('bertscore') with
    rescale_with_baseline=True. Can be disabled on memory-constrained hosts via
    DISABLE_BERTSCORE=1 — the honest fallback is an explicit zero, never a fake score.
    """
    if os.getenv('DISABLE_BERTSCORE', '').lower() in ('1', 'true', 'yes'):
        return {'raw_f1': 0.0, 'rescaled_f1': 0.0, 'bonus_hit': False, 'disabled': True}

    global _bertscore_metric
    try:
        if _bertscore_metric is None:
            import evaluate
            _bertscore_metric = evaluate.load('bertscore')
        model_type = os.getenv('BERTSCORE_MODEL', 'distilbert-base-uncased')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rescaled = _bertscore_metric.compute(
                predictions=predictions, references=references,
                lang='en', model_type=model_type, rescale_with_baseline=True,
            )
            raw = _bertscore_metric.compute(
                predictions=predictions, references=references,
                lang='en', model_type=model_type,
            )
        rescaled_f1 = sum(rescaled['f1']) / len(rescaled['f1'])
        raw_f1      = sum(raw['f1']) / len(raw['f1'])
        return {
            'raw_f1':      round(raw_f1, 4),
            'rescaled_f1': round(rescaled_f1, 4),
            'bonus_hit':   rescaled_f1 >= 0.55 or raw_f1 >= 0.88,
        }
    except Exception as e:
        return {'raw_f1': 0.0, 'rescaled_f1': 0.0, 'bonus_hit': False, 'error': str(e)}
