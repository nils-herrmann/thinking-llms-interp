# %%
import dotenv
dotenv.load_dotenv("../.env")

# Standard lib
import argparse
import json
import os
import random
import re
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

# Third-party
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # add cot-interp/ to path
import utils  # noqa: E402
from utils import steering_opt  # noqa: E402
from utils.utils import split_into_sentences, get_char_to_token_map, convert_numpy_types  # noqa: E402
from utils.utils import load_steering_vectors  # noqa: E402
from utils.clustering import run_chat_batch_with_event_loop_handling  # noqa: E402
from utils.utils import chat_batch  # noqa: E402


# ---------------------------------------------------------------------
# Helper: safe_chat_batch – identical event-loop handling but allows
# overriding `max_tokens` so we don’t exceed model limits.
# ---------------------------------------------------------------------

def safe_chat_batch(prompts, model_name: str, max_tokens: int = 1024, **kwargs):
    """Synchronous wrapper around utils.utils.chat_batch with smaller max_tokens."""
    import asyncio
    import concurrent.futures

    async def _run():
        return await chat_batch(
            prompts,
            model=model_name,
            max_tokens=max_tokens,
            **kwargs,
        )

    try:
        loop = asyncio.get_running_loop()

        def _thread_runner():
            return asyncio.run(_run())

        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(_thread_runner)
            return fut.result()
    except RuntimeError:
        # no running loop
        return asyncio.run(_run())

# =============================================================
# 1.  SAE annotation helpers  (largely copied from optimise script)
# =============================================================

ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
CATEGORY_PATTERN = re.compile(r'\["[\d.]+:(\S+?)"\]')


def get_label_positions(annotated_thinking: str,
                        response_text: str,
                        tokenizer,
                        context_sentences: int = 0) -> Dict[str, List[Tuple[int, int, str, float, int]]]:
    """Parse SAE annotations and find token positions for each label.

    Returns a mapping  label -> list[(token_start, token_end, text, activation, char_pos)].
    """
    label_positions: Dict[str, List[Tuple[int, int, str, float, int]]] = {}
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))

    # map char -> token once for efficiency
    char_to_token = get_char_to_token_map(response_text, tokenizer)
    sentences = split_into_sentences(response_text, min_words=0)

    for match in matches[:-1]:
        activation_str, label, text = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        try:
            activation = float(activation_str)
        except ValueError:
            continue
        if not text:
            continue

        # locate the *first* occurrence of this exact text fragment in the full response
        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        m = re.search(pattern, response_text)
        if not m:
            continue
        text_pos = m.start(1)

        token_start = char_to_token.get(text_pos)
        token_end = char_to_token.get(text_pos + len(text) - 1)
        if token_end is not None:
            token_end += 1  # make slice end-exclusive
        if None in (token_start, token_end) or token_start >= token_end:
            continue

        # identify which sentence the text lives in so we can append extra context sentences
        target_sentence_idx = next((i for i, s in enumerate(sentences) if text in s), -1)
        if target_sentence_idx == -1:
            continue

        additional_context = ""
        if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
            end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
            additional_sentences = sentences[target_sentence_idx + 1 : end_idx]
            text_end_pos = text_pos + len(text)
            next_sentence_start = response_text.find(additional_sentences[0], text_end_pos) if additional_sentences else -1
            if next_sentence_start > text_end_pos:
                ws = response_text[text_end_pos : next_sentence_start]
                additional_context = ws + ws.join(additional_sentences)
            elif additional_sentences:
                additional_context = " " + " ".join(additional_sentences)

            if additional_context:
                ctx_end_char = text_pos + len(text) + len(additional_context) - 1
                ctx_end_tok = char_to_token.get(ctx_end_char)
                if ctx_end_tok is not None:
                    token_end = ctx_end_tok + 1

        label_positions.setdefault(label, []).append((token_start, token_end, text + additional_context, activation, text_pos))

    return label_positions


# =============================================================
# 2.  Example extraction with activation/perplexity ranking
# =============================================================

def extract_eval_examples_for_category(
    responses_data: List[dict],
    category_name: str,
    tokenizer,
    model,
    n_eval_examples: int = 64,
    context_sentences: int = 0,
    n_training_examples: int = 8,  # Default value, will be overridden by hyperparams
) -> List[dict]:
    """Return ≤ *n_eval_examples* unseen examples for *category_name* following the
    activation→perplexity selection procedure.
    """
    # (1) gather *all* candidate examples for the category ----------------------------------
    examples: List[dict] = []
    for resp in responses_data:
        if not resp.get("annotated_thinking"):
            continue
        full_text = (
            "Task: Answer the question below. Explain your reasoning step by step.\n\n\n\n"
            f"Question:\n{resp['original_message']['content']}\n\nStep by step answer:\n{resp['thinking_process']}"
        )
        if category_name not in resp["annotated_thinking"]:
            continue

        label_positions = get_label_positions(resp["annotated_thinking"], full_text, tokenizer, context_sentences)
        if category_name not in label_positions:
            continue

        for start, end, text, activation, text_pos in label_positions[category_name]:
            context = full_text[:text_pos]
            if not context.strip():
                continue
            # crude check that context ends on boundary
            if context[-1] not in [".", "?", "!", ";", "\n"]:
                continue

            word_count = len(text.strip().split())
            if word_count < 10:
                continue
        
            examples.append(
                {
                    "prompt": context,
                    "target_completion": text,
                    "original_question": resp["original_message"]["content"],
                    "full_thinking": resp["thinking_process"],
                    "activation": activation,
                }
            )

    if not examples:
        return []

    # (2) Sort by activation and select range after training examples ----------------------
    examples = sorted(examples, key=lambda x: x["activation"], reverse=True)
    start_idx = min(8 * n_training_examples, len(examples) - 4 * n_eval_examples)
    end_idx = min(start_idx + 4 * n_eval_examples, len(examples))
    candidate_examples = examples[start_idx:end_idx]

    if not candidate_examples:
        # Fallback to random sampling if we don't have enough examples
        return random.sample(examples, min(n_eval_examples, len(examples)))

    # (3) Calculate perplexity for the candidate examples --------------------------------
    scored: List[Tuple[float, dict]] = []
    for ex in candidate_examples:
        try:
            txt = ex["prompt"] + ex["target_completion"]
            inputs = tokenizer(txt, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits[0]
            prompt_tok = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"][0]
            prompt_len = len(prompt_tok)
            shift_logits = logits[:-1, :].contiguous()
            shift_labels = inputs["input_ids"][0][1:].contiguous()
            target_logits = shift_logits[prompt_len - 1 :]
            target_labels = shift_labels[prompt_len - 1 :]
            loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction="mean")
            perplexity = torch.exp(loss).item()
            scored.append((perplexity, ex))
        except Exception:
            continue

    # Sort by perplexity (descending) and take top n_eval_examples
    scored = sorted(scored, key=lambda t: t[0], reverse=True)[:n_eval_examples]
    selected = [ex for _pp, ex in scored]

    # Remove perplexity/activation from the final examples
    for ex in selected:
        ex.pop('activation', None)

    return selected


# =============================================================
# 3.  Steering & generation helper
# =============================================================

def generate_steered_completion(
    model,
    tokenizer,
    vector: torch.Tensor,
    layer: int,
    prompt: str,
    target_completion: str,
    max_new_tokens: int = 100,
    steering_token_window: Optional[int] = 50,
) -> str:
    """Generate continuation with steering vector applied from the first target token."""
    prompt_tok = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    target_tok = tokenizer(target_completion, return_tensors="pt")["input_ids"][0]
    prompt_len, target_len = len(prompt_tok), len(target_tok)

    steering_start = (
        prompt_len
        if steering_token_window is None
        else prompt_len + max(0, target_len - steering_token_window)
    )
    steering_slice = slice(steering_start, None)
    hook = (layer, steering_opt.make_steering_hook_hf(vector, token=steering_slice))

    with steering_opt.hf_hooks_contextmanager(model, [hook]):
        gen_tokens = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return full[len(prompt) :]

# ---------------------------------------------------------------------
# Helper: return the last *n* tokens of a text string
# ---------------------------------------------------------------------

def get_last_tokens(text: str, tokenizer, n: int = 100) -> str:
    """Return decoded string of the last *n* tokens of *text*."""
    toks = tokenizer(text, return_tensors="pt")["input_ids"][0]
    if len(toks) <= n:
        return text
    return tokenizer.decode(toks[-n:], skip_special_tokens=True)

# =============================================================
# 4.  Category description loader (from clustering results)
# =============================================================

def load_category_descriptions(model_id: str, layer: int) -> Dict[str, str]:
    """Scan clustering results JSON files and build mapping short-title-lower -> description.

    We look for files matching `*_results_{model_id}_layer{layer}.json` inside
    `cot-interp/train-saes/results/vars`.
    """
    root = os.path.join(os.path.dirname(__file__), "..", "train-saes", "results", "vars")
    descs: Dict[str, str] = {}
    if not os.path.isdir(root):
        return descs

    for fn in os.listdir(root):
        if not fn.endswith(f"_{model_id}_layer{layer}.json"):
            continue
        try:
            with open(os.path.join(root, fn), "r") as f:
                data = json.load(f)
        except Exception:
            continue
        det = data.get("detailed_results", {})
        for cl_id, entry in det.items():
            title = entry.get("title", "").lower()
            desc = entry.get("description", "")
            if title and desc:
                descs[title] = desc
    return descs


def match_description(category_name: str, descriptions: Dict[str, str]) -> Optional[str]:
    """Very simple heuristic string match to find description for *category_name*."""
    key = category_name.replace("-", " ").replace("/", " ").lower()
    for title, desc in descriptions.items():
        if all(tok in title for tok in key.split()):
            return desc
    return None


# =============================================================
# 5.  Autograder prompt & scoring helpers
# =============================================================

def build_autograder_prompt(category: str, description: str, target: str, generated: str) -> str:
    """Create an evaluation prompt for the autograder LLM."""
    return f'''# Task: Evaluate steered completion for reasoning category

You are an expert evaluator of fine-grained reasoning functions exhibited in large-language-model chain-of-thought traces.
The target category is:

**{category}**

Description:
{description}

Below you will find two text segments:
1. A *ground-truth* example that cleanly exemplifies the target category.
2. A *steered completion* produced by the base model after applying a steering vector intended to induce the same category.

---
GROUND TRUTH EXAMPLE (reference):
"""
{target.strip()}
"""

STEERED COMPLETION (candidate):
"""
{generated.strip()}
"""
---

Please rate *how strongly* the **candidate** demonstrates the target reasoning function *relative to* the reference.

Guidelines:
* 1.0 = clearly and unambiguously exhibits the exact same reasoning behaviour.
* 0.0 = does **not** exhibit the reasoning function at all (or is irrelevant / hallucinated / etc.).
* Intermediate values represent partial alignment (e.g. 0.5 = somewhat exhibits but with gaps).

Respond **only** with a JSON object in the form:
```json
{{"score": <float between 0.0 and 1.0>}}
```
Do not include any additional keys or commentary.
'''


def parse_score(response: str) -> Optional[float]:
    try:
        obj = json.loads(response.strip().split("```json")[-1].split("```")[-2]) if "```json" in response else json.loads(response)
        s = float(obj.get("score"))
        if 0.0 <= s <= 1.0:
            return s
    except Exception:
        pass
    return None


# =============================================================
# 6.  Main
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate steering vectors with autograder")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--vectors_dir", type=str, default="results/vars/optimized_vectors")
    parser.add_argument("--hyperparams_dir", type=str, default="results/vars/hyperparams", help="Directory with per-vector hyperparameter JSON files")
    parser.add_argument("--n_eval_examples", type=int, default=16)
    parser.add_argument("--test_max_tokens", type=int, default=80)
    parser.add_argument("--context_sentences", type=int, default=0)
    parser.add_argument("--steering_token_window", type=int, default=80)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autograder_model", type=str, default="openai/gpt-4o")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of independent evaluation runs (for CI)")
    args, _ = parser.parse_known_args()

    # Aggregate results across runs (separate steered vs. baseline)
    aggregated_steered_scores: Dict[str, List[float]] = {}
    aggregated_baseline_scores: Dict[str, List[float]] = {}

    # perform multiple runs as requested
    for run_idx in range(args.n_runs):
        print(f"\n######################\n# RUN {run_idx + 1}/{args.n_runs}\n######################")

        # vary seed per run for slight diversity
        run_seed = args.seed + run_idx
        random.seed(run_seed)
        torch.manual_seed(run_seed)

        # -------------------------- Model & tokenizer -------------------------- #
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            torch_dtype=torch.bfloat16,
        )
        torch.set_default_device(base_model.device)
        for p in base_model.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------
        # Load ALL steering vectors for this model using the new util
        # ------------------------------------------------------------
        hyperparams_dir_abs = os.path.join(os.path.dirname(__file__), args.hyperparams_dir)
        vectors_dir_abs = os.path.join(os.path.dirname(__file__), args.vectors_dir)

        vectors_map = load_steering_vectors(
            device=base_model.device,
            hyperparams_dir=hyperparams_dir_abs,
            vectors_dir=vectors_dir_abs,
            verbose=False,
        )

        all_hparams = {}
        model_short = args.model.split("/")[-1].lower()
        model_hparams = all_hparams.get(model_short, {})

        # -------------------------- Load responses & annotations --------------- #
        thinking_model_name = utils.model_mapping.get(args.model, model_short)
        thinking_short = thinking_model_name.split("/")[-1].lower()
        responses_path = os.path.join(os.path.dirname(__file__), "..", "generate-responses", "results", "vars", f"responses_{thinking_short}.json")
        annotated_path = os.path.join(os.path.dirname(__file__), "..", "generate-responses", "results", "vars", f"annotated_responses_{thinking_short}.json")
        if not os.path.exists(responses_path) or not os.path.exists(annotated_path):
            raise FileNotFoundError("Responses or annotated responses not found – please generate & annotate first.")
        with open(responses_path, "r") as f:
            responses_data = json.load(f)
        with open(annotated_path, "r") as f:
            annotated_data = json.load(f)

        valid_responses = []
        for i, resp in enumerate(responses_data):
            if i < len(annotated_data):
                ann = annotated_data[i]
                if (
                    resp.get("question_id") == ann.get("question_id")
                    and resp.get("dataset_name") == ann.get("dataset_name")
                    and ann.get("annotated_thinking")
                ):
                    merged = resp.copy()
                    merged["annotated_thinking"] = ann["annotated_thinking"]
                    valid_responses.append(merged)
        if not valid_responses:
            raise RuntimeError("No valid annotated responses found.")

        # -------------------------- Category descriptions ---------------------- #
        # We fetch *once* for the layer used by (most) vectors (assume consistent)
        # If per-vector layers vary we fallback to no description.
        sample_layer = next(iter(model_hparams.values()))["hyperparameters"]["layer"] if model_hparams else 0
        category_descs = load_category_descriptions(model_short.replace("-", "_"), sample_layer)

        # -------------------------- Iterate over steering vectors -------------- #
        results = {}
        tokens_output = {}

        hp_dir_abs = os.path.join(os.path.dirname(__file__), args.hyperparams_dir)

        for hp_file in os.listdir(hp_dir_abs):
            if not (
                hp_file.startswith(f"steering_vector_hyperparams_{model_short}_") and hp_file.endswith(".json")
            ):
                continue

            try:
                idx = int(hp_file.split(f"steering_vector_hyperparams_{model_short}_")[-1].split(".")[0])
            except Exception:
                continue

            # Load hyperparameters for this vector
            try:
                with open(os.path.join(hp_dir_abs, hp_file), "r") as f:
                    hp_entry = json.load(f)
            except Exception:
                print(f"[WARN] Could not read {hp_file} – skipping.")
                continue

            category = hp_entry.get("category")
            if not category:
                continue

            # Get n_training_examples from hyperparameters
            n_training_examples = hp_entry.get("hyperparameters", {}).get("n_training_examples", 8)
            layer = hp_entry.get("hyperparameters", {}).get("layer", 0)

            vector = vectors_map.get(category)
            if vector is None:
                print(f"[WARN] Vector for category '{category}' not found – skipping.")
                continue

            vector = vector.to(base_model.device).to(base_model.dtype)

            # description
            desc = match_description(category, category_descs) or "(No description available)"

            print(f"\n=== Evaluating idx {idx} — {category} (layer {layer}) ===")

            eval_examples = extract_eval_examples_for_category(
                valid_responses,
                category,
                tok,
                base_model,
                n_eval_examples=args.n_eval_examples,
                context_sentences=args.context_sentences,
                n_training_examples=n_training_examples
            )
            if not eval_examples:
                print("  -> No evaluation examples found – skipping.")
                continue

            prompts_steered = []
            prompts_baseline = []
            eval_example_outputs = []

            for ex in tqdm(eval_examples, desc="Generating steered & baseline completions"):
                # ---------------- Steered completion ----------------
                steered_gen = generate_steered_completion(
                    base_model,
                    tok,
                    vector,
                    layer,
                    ex["prompt"],
                    ex["target_completion"],
                    max_new_tokens=args.test_max_tokens,
                    steering_token_window=args.steering_token_window,
                )

                # ---------------- Baseline completion (no steering) ----------------
                with torch.no_grad():
                    baseline_tokens = base_model.generate(
                        **tok(ex["prompt"], return_tensors="pt").to(base_model.device),
                        max_new_tokens=args.test_max_tokens,
                        pad_token_id=tok.eos_token_id,
                    )
                baseline_gen = tok.batch_decode(baseline_tokens, skip_special_tokens=True)[0][len(ex["prompt"]):]

                # Build autograder prompts
                prompts_steered.append(build_autograder_prompt(category, desc, ex["target_completion"], steered_gen))
                prompts_baseline.append(build_autograder_prompt(category, desc, ex["target_completion"], baseline_gen))

                # Store last 100 tokens of various strings
                eval_example_outputs.append(
                    {
                        "input_last_tokens": get_last_tokens(ex["prompt"], tok, 100),
                        "target_last_tokens": get_last_tokens(ex["target_completion"], tok, 100),
                        "steered_last_tokens": get_last_tokens(steered_gen, tok, 100),
                        "baseline_last_tokens": get_last_tokens(baseline_gen, tok, 100),
                    }
                )

            # ---------------- Autograde steered completions ----------------
            print("Sending steered prompts to autograder…")
            responses_steered = safe_chat_batch(prompts_steered, args.autograder_model, max_tokens=1024)
            scores_steered = [parse_score(r) for r in responses_steered]
            valid_scores_steered = [s for s in scores_steered if s is not None]
            avg_steered = sum(valid_scores_steered) / len(valid_scores_steered) if valid_scores_steered else 0.0
            print(
                f"Average *steered* autograder score over {len(valid_scores_steered)}/{len(prompts_steered)} examples: {avg_steered:.3f}"
            )

            # ---------------- Autograde baseline completions ----------------
            print("Sending baseline prompts to autograder…")
            responses_baseline = safe_chat_batch(prompts_baseline, args.autograder_model, max_tokens=1024)
            scores_baseline = [parse_score(r) for r in responses_baseline]
            valid_scores_baseline = [s for s in scores_baseline if s is not None]
            avg_baseline = sum(valid_scores_baseline) / len(valid_scores_baseline) if valid_scores_baseline else 0.0
            print(
                f"Average *baseline* autograder score over {len(valid_scores_baseline)}/{len(prompts_baseline)} examples: {avg_baseline:.3f}"
            )

            # attach scores back to the example records
            for i in range(len(eval_example_outputs)):
                if i < len(scores_steered):
                    eval_example_outputs[i]["steered_score"] = scores_steered[i]
                if i < len(scores_baseline):
                    eval_example_outputs[i]["baseline_score"] = scores_baseline[i]

            results[category] = {
                "idx": idx,
                "layer": layer,
                "avg_score_steered": avg_steered,
                "avg_score_baseline": avg_baseline,
                "scores_steered": valid_scores_steered,
                "scores_baseline": valid_scores_baseline,
                "n_examples": len(eval_examples),
            }

            # collect token-level data for this category
            tokens_output[category] = eval_example_outputs

        # -------------------------- Save per-run results ------------------ #
        out_dir = os.path.join(os.path.dirname(__file__), "results", "vars")
        os.makedirs(out_dir, exist_ok=True)

        run_out_path = os.path.join(out_dir, f"steering_vector_eval_scores_run{run_idx + 1}.json")
        with open(run_out_path, "w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        print(f"Saved run {run_idx + 1} scores to {run_out_path}")

        # merge into aggregated lists
        for cat, det in results.items():
            aggregated_steered_scores.setdefault(cat, []).append(det["avg_score_steered"])
            aggregated_baseline_scores.setdefault(cat, []).append(det["avg_score_baseline"])

        # Save last-token data only for first run (to limit size)
        if run_idx == 0:
            tokens_path = os.path.join(out_dir, "steering_vector_last_tokens.json")
            with open(tokens_path, "w") as f:
                json.dump(convert_numpy_types(tokens_output), f, indent=2)
            print(f"Saved last-token data to {tokens_path}")

    # -------------------------- After all runs: compute CI & plot --------- #
    if aggregated_steered_scores:
        # sort categories by steered mean for consistent ordering
        cats = sorted(aggregated_steered_scores.keys(), key=lambda k: np.mean(aggregated_steered_scores[k]), reverse=True)

        steered_means = [float(np.mean(aggregated_steered_scores[c])) for c in cats]
        baseline_means = [float(np.mean(aggregated_baseline_scores[c])) for c in cats]

        steered_cis = [
            float(1.96 * np.std(aggregated_steered_scores[c], ddof=1) / np.sqrt(len(aggregated_steered_scores[c])))
            if len(aggregated_steered_scores[c]) > 1
            else 0.0
            for c in cats
        ]
        baseline_cis = [
            float(1.96 * np.std(aggregated_baseline_scores[c], ddof=1) / np.sqrt(len(aggregated_baseline_scores[c])))
            if len(aggregated_baseline_scores[c]) > 1
            else 0.0
            for c in cats
        ]

        # save aggregated json (both steered & baseline)
        agg_path = os.path.join(out_dir, "steering_vector_eval_scores_runs.json")
        with open(agg_path, "w") as f:
            json.dump(
                {
                    c: {
                        "steered": {"mean": sm, "ci_95": sc, "runs": aggregated_steered_scores[c]},
                        "baseline": {"mean": bm, "ci_95": bc, "runs": aggregated_baseline_scores[c]},
                    }
                    for c, sm, sc, bm, bc in zip(cats, steered_means, steered_cis, baseline_means, baseline_cis)
                },
                f,
                indent=2,
            )
        print(f"Saved aggregated scores to {agg_path}")

        # grouped bar chart with error bars
        x = np.arange(len(cats))
        width = 0.35
        plt.figure(figsize=(14, 6))
        plt.bar(x - width / 2, baseline_means, width, yerr=baseline_cis, capsize=5, label="Baseline")
        plt.bar(x + width / 2, steered_means, width, yerr=steered_cis, capsize=5, label="Steered")
        plt.ylabel("Average Autograder Score")
        plt.xlabel("Category")
        plt.title(f"Steered vs Baseline Autograder Scores (n_runs={args.n_runs}) with 95% CI")
        plt.xticks(x, cats, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        chart_path = os.path.join("./results/figures/steering_vs_baseline_eval_scores_CI.png")
        plt.savefig(chart_path)
        print(f"Saved CI grouped bar chart to {chart_path}")


if __name__ == "__main__":
    main()

# %% 