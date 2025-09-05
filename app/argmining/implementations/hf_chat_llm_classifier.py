import re
import os
import torch
from typing import List, Optional
from uuid import uuid4, UUID

from transformers import AutoTokenizer, AutoModelForCausalLM

from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import (
    ArgumentUnit,
    LinkedArgumentUnits,
    LinkedArgumentUnitsWithStance,
    StanceRelation,
    UnlinkedArgumentUnits,
)

# Try relative import first (for benchmark), then app logger
try:
    from ...log import log
except ImportError:
    from app.log import log as _log
    def log():
        return _log

from ..config import HF_TOKEN


def split_into_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())


class HFChatLLMClassifier(AduAndStanceClassifier):
    """
    Generic HuggingFace chat-tuned LLM classifier with constrained decoding.
    Works with models that provide a chat template (e.g., Llama, Qwen, Phi).
    """

    def __init__(self, base_model_id: str, name: Optional[str] = None):
        self.base_model_id = base_model_id
        self.name = name or base_model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if HF_TOKEN:
            log().info(f"HF_TOKEN is configured (length: {len(HF_TOKEN)})")
        else:
            log().warning("HF_TOKEN is not configured - model download may fail if needed")

        # Tokenizer + chat template
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=HF_TOKEN)
        self._has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)

        # Model
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map=None if self.device == "cpu" else "auto",
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            base_model = base_model.to(self.device)
            # Reduce KV cache memory on CPU
            try:
                base_model.config.use_cache = False
            except Exception:
                pass
        self.model = base_model

        # Prompts
        self.system_prompt_adu = (
            "You are an argument-mining classifier.\n\n"
            "Task: Decide whether the TARGET sentence is a claim or a premise.\n\n"
            "Definitions (use these only):\n"
            "- claim: a statement that takes a stance or asserts something to be true/false or should/shouldn't happen.\n"
            "- premise: a statement that gives evidence, reasons, data, or explanation intended to support/refute a claim.\n\n"
            "Rules:\n"
            "- You will be given a TARGET sentence and surrounding context sentences.\n"
            "- Focus on classifying the TARGET sentence, but use the context to understand its role in the argument.\n"
            "- Consider how the TARGET sentence relates to nearby sentences (does it support them, or do they support it?).\n"
            "- Output EXACTLY ONE lowercase word: claim or premise.\n"
            "- If the sentence mixes both, pick the main function.\n"
        )
        self.system_prompt_stance = (
            "You are an assistant for argument mining.\n"
            "You are given a claim and evidence (premise) along with their surrounding context.\n"
            "Determine whether the evidence supports ('pro') or refutes ('con') the claim.\n"
            "Use the context to better understand the relationship between claim and premise.\n"
            "Respond only with one word: pro or con.\n"
        )
        # Optional few-shot examples
        self.few_shot_adu_examples = (
            "Examples:\n"
            "Context before: \"Many cities invest in public transit.\"\n"
            "TARGET sentence: \"Subsidizing buses reduces traffic congestion.\"\n"
            "Context after: \"This policy also lowers emissions.\"\n"
            "Answer: premise\n\n"
            "Context before: \"A new tax was proposed last year.\"\n"
            "TARGET sentence: \"The tax should be implemented immediately.\"\n"
            "Context after: \"Opponents argue it harms small businesses.\"\n"
            "Answer: claim\n"
        )
        self.few_shot_stance_examples = (
            "Examples:\n"
            "Claim with context: Banning plastic bags will protect the environment.\n"
            "Evidence with context: Studies show a reduction in litter after bans.\n"
            "Stance: pro\n\n"
            "Claim with context: The new speed cameras are unnecessary.\n"
            "Evidence with context: Data indicates accidents decreased without cameras.\n"
            "Stance: con\n"
        )

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        if self._has_chat_template:
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n\nAnswer:"

    def _run_prompt(self, prompt: str, max_new_tokens: int, allowed_responses: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        prefix_allowed_tokens_fn = None
        if allowed_responses:
            candidates = []
            for r in allowed_responses:
                for variant in (r, " " + r):
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    if ids:
                        candidates.append(ids)

            def _prefix_allowed_tokens_fn(batch_id, input_ids_row):
                gen = input_ids_row[input_len:]
                pos = len(gen)
                allowed = set()
                for seq in candidates:
                    if pos < len(seq) and (pos == 0 or list(gen[:pos].tolist()) == seq[:pos]):
                        allowed.add(seq[pos])
                if pos and any(len(seq) == pos and list(gen[:pos].tolist()) == seq for seq in candidates):
                    allowed.add(self.tokenizer.eos_token_id)
                if not allowed and pos == 0:
                    for seq in candidates:
                        allowed.add(seq[0])
                if not allowed:
                    allowed.add(self.tokenizer.eos_token_id)
                return list(allowed)

            prefix_allowed_tokens_fn = _prefix_allowed_tokens_fn

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                # Silence sampling-related warnings for deterministic decoding
                temperature=None,
                top_p=None,
                top_k=None,
            )
        gen_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Interface methods
    def get_context_window(self, sentences: List[str], target_index: int, window_size: int = 2) -> tuple[str, str, str]:
        start_idx = max(0, target_index - window_size)
        end_idx = min(len(sentences), target_index + window_size + 1)
        before_context = " ".join(sentences[start_idx:target_index]) if target_index > 0 else ""
        target_sentence = sentences[target_index]
        after_context = " ".join(sentences[target_index + 1:end_idx]) if target_index < len(sentences) - 1 else ""
        return before_context, target_sentence, after_context

    def classify_sentence_with_context(self, sentences: List[str], target_index: int, use_few_shot: bool | None = None) -> str:
        before_context, target_sentence, after_context = self.get_context_window(sentences, target_index)
        user_prompt = (
            ((self.few_shot_adu_examples + "\n\n") if use_few_shot else "") +
            f"Context before: \"{before_context}\"\n"
            f"TARGET sentence: \"{target_sentence}\"\n"
            f"Context after: \"{after_context}\"\n\n"
            "Is the TARGET sentence a claim or a premise? Respond with exactly one word: claim or premise."
        )
        full_prompt = self._build_prompt(self.system_prompt_adu, user_prompt)
        response = self._run_prompt(full_prompt, max_new_tokens=4, allowed_responses=["claim", "premise"])
        m = re.search(r"\b(claim|premise)\b", response.lower())
        if m:
            return m.group(1)
        log().warning(f"Unexpected ADU classification output: {response} - labeling as 'unknown'")
        return "unknown"

    def classify_sentence(self, sentence: str, use_few_shot: bool | None = None) -> str:
        return self.classify_sentence_with_context([sentence], 0, use_few_shot)

    def classify_adus(self, text: str, use_few_shot: bool | None = None) -> UnlinkedArgumentUnits:
        sentences = split_into_sentences(text)
        log().info(f"[{self.name}] Found {len(sentences)} sentences in the input text")
        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []
        current_pos = 0
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            adu_type = self.classify_sentence_with_context(sentences, i, use_few_shot)
            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence) if start_pos != -1 else -1
            if start_pos != -1:
                current_pos = end_pos
            if adu_type not in ("claim", "premise"):
                log().warning(f"Skipping sentence due to uncertain ADU type: '{sentence}' → {adu_type}")
                continue
            unit = ArgumentUnit(
                uuid=uuid4(),
                text=sentence,
                type=adu_type,
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=None,
            )
            (claims if adu_type == "claim" else premises).append(unit)
        return UnlinkedArgumentUnits(claims=claims, premises=premises)

    def classify_stance_single(self, claim_text: str, premise_text: str, original_text: Optional[str] = None, use_few_shot: bool | None = None) -> str:
        if original_text:
            sentences = split_into_sentences(original_text)
            claim_context = claim_text
            premise_context = premise_text
            # Basic context expansion
            for s in sentences:
                if claim_text.strip() in s:
                    claim_context = s
                if premise_text.strip() in s:
                    premise_context = s
            user_prompt = (
                f"Claim with context: {claim_context}\n"
                f"Evidence with context: {premise_context}\n\n"
                f"Focus on this specific claim: \"{claim_text}\"\n"
                f"Focus on this specific evidence: \"{premise_text}\"\n\n"
                "Stance:"
            )
        else:
            user_prompt = f"Claim: {claim_text}\nEvidence: {premise_text}\nStance:"
        sys_prompt = self.system_prompt_stance
        if use_few_shot:
            sys_prompt = (sys_prompt + "\n\n" + self.few_shot_stance_examples)
        full_prompt = self._build_prompt(sys_prompt, user_prompt)
        response = self._run_prompt(full_prompt, max_new_tokens=4, allowed_responses=["pro", "con"])
        m = re.search(r"\b(pro|con|support|refute)\b", response.lower())
        if m:
            tok = m.group(1)
            return "con" if tok in ("con", "refute") else "pro"
        log().warning(f"Unexpected stance output: {response} for Claim: '{claim_text}' | Premise: '{premise_text}'")
        return "unidentified"

    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str, use_few_shot: bool | None = None) -> LinkedArgumentUnitsWithStance:
        result: List[StanceRelation] = []
        for rel in linked_argument_units.claims_premises_relationships:
            claim = next((c for c in linked_argument_units.claims if c.uuid == rel.claim_id), None)
            if not claim:
                continue
            if not rel.premise_ids:
                log().warning(f"No premises found for claim '{claim.text}' ---> Continuing loop")
                continue
            for pid in rel.premise_ids:
                premise = next((p for p in linked_argument_units.premises if p.uuid == pid), None)
                if not premise:
                    continue
                stance = self.classify_stance_single(claim.text, premise.text, original_text=originalText, use_few_shot=use_few_shot)
                result.append(
                    StanceRelation(
                        claim_id=claim.uuid,
                        premise_id=premise.uuid,
                        stance=stance,
                        confidence=None,
                    )
                )
        return LinkedArgumentUnitsWithStance(
            original_text=originalText,
            claims=linked_argument_units.claims,
            premises=linked_argument_units.premises,
            stance_relations=result,
        )
