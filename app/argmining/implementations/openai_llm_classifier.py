from dataclasses import asdict
import json
import openai
import re 
import os

from uuid import UUID, uuid4

from .openai_claim_premise_linker import OpenAIClaimPremiseLinker
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, ClaimPremiseRelationship, UnlinkedArgumentUnits
from typing import List
from app.log import logger
from app.argmining.config import OPENAI_KEY


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())
      
class OpenAILLMClassifier(AduAndStanceClassifier): 
    def __init__(self, model_name: str = "gpt-4.1"): 
        self.client = openai.OpenAI(api_key=OPENAI_KEY)
        self.model_name = model_name  # Store the specific OpenAI model to use
        self.system_prompt_adu_classification = """You are an argument-mining classifier.

Task: Decide whether the TARGET sentence is a **claim** or a **premise**.

Definitions (use these only):
- claim: a statement that takes a stance or asserts something to be true/false or should/shouldn't happen.
- premise: a statement that gives evidence, reasons, data, or explanation intended to support/refute a claim.

Rules:
- You will be given a TARGET sentence and surrounding context sentences.
- Focus on classifying the TARGET sentence, but use the context to understand its role in the argument.
- Consider how the TARGET sentence relates to nearby sentences (does it support them, or do they support it?).
- Output EXACTLY ONE lowercase word: "claim" or "premise".
- If the sentence mixes both, pick the main function (assertion → claim; support/explanation → premise).
- Do NOT add punctuation or extra text.

Answer:
                        """
        
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
        self.system_prompt_stance_classification = """You are an assistant for argument mining.
You are given a claim and evidence (premise) along with their surrounding context.
Determine whether the evidence supports ('pro') or refutes ('con') the claim.
Use the context to better understand the relationship between claim and premise.
Respond only with one word: "pro" or "con".
                        """
        
    def get_context_window(self, sentences: List[str], target_index: int, window_size: int = 2) -> tuple[str, str, str]:
        start_idx = max(0, target_index - window_size)
        end_idx = min(len(sentences), target_index + window_size + 1)
        
        before_context = " ".join(sentences[start_idx:target_index]) if target_index > 0 else ""
        target_sentence = sentences[target_index]
        after_context = " ".join(sentences[target_index + 1:end_idx]) if target_index < len(sentences) - 1 else ""
        
        return before_context, target_sentence, after_context
        
    def classify_sentence_with_context(self, sentences: List[str], target_index: int, model: str = None, use_few_shot: bool | None = None) -> str:
        if model is None:
            model = self.model_name  # Use instance model if not specified
        before_context, target_sentence, after_context = self.get_context_window(sentences, target_index)
        
        user_prompt = (
            (self.few_shot_adu_examples + "\n\n") if use_few_shot else ""
        ) + f"""Context before: "{before_context}"
TARGET sentence: "{target_sentence}"
Context after: "{after_context}"

Is the TARGET sentence a claim or a premise? Respond with exactly one word: claim or premise."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt_adu_classification.strip(),
                    },
                    {"role": "user", "content": user_prompt.strip()},
                ],
                # No temperature param (some models only allow default)
            )
            result = response.choices[0].message.content.strip().lower()  # type: ignore
            
            if "claim" in result:
                return "claim"
            elif "premise" in result:
                return "premise"
            else:
                logger.warning(f"⚠️ Unexpected ADU classification output: {result} - labeling as 'unknown'")
                return "unknown"
        except Exception as e:
            logger.warning(f"❌ Model {model} failed for ADU classification: {e}")
            if model != "gpt-3.5-turbo":
                logger.info("Attempting with gpt-3.5-turbo for ADU classification.")
                return self.classify_sentence_with_context(sentences, target_index, model="gpt-3.5-turbo")
            logger.error("Failed ADU classification after retries. Returning 'unknown'.")
            return "unknown"  # Fallback if all models fail

    def classify_sentence(self, sentence: str, model: str = None, use_few_shot: bool | None = None) -> str:
        return self.classify_sentence_with_context([sentence], 0, model, use_few_shot)
        
    def find_context_for_units(self, text: str, claim_unit: ArgumentUnit, premise_unit: ArgumentUnit, window_size: int = 1) -> tuple[str, str]:
        sentences = split_into_sentences(text)
        
        claim_sentence_idx = -1
        premise_sentence_idx = -1
        
        for i, sentence in enumerate(sentences):
            if claim_unit.text.strip() in sentence:
                claim_sentence_idx = i
            if premise_unit.text.strip() in sentence:
                premise_sentence_idx = i
                
        if claim_sentence_idx >= 0:
            start_idx = max(0, claim_sentence_idx - window_size)
            end_idx = min(len(sentences), claim_sentence_idx + window_size + 1)
            claim_context = " ".join(sentences[start_idx:end_idx])
        else:
            claim_context = claim_unit.text
            
        if premise_sentence_idx >= 0:
            start_idx = max(0, premise_sentence_idx - window_size)
            end_idx = min(len(sentences), premise_sentence_idx + window_size + 1)
            premise_context = " ".join(sentences[start_idx:end_idx])
        else:
            premise_context = premise_unit.text
            
        return claim_context, premise_context
        
    def classify_stance_single(self, claim_text: str, premise_text: str, model: str = None, original_text: str = None, use_few_shot: bool | None = None) -> str:
        if model is None:
            model = self.model_name  # Use instance model if not specified
        if original_text:
            claim_unit = ArgumentUnit(uuid=uuid4(), text=claim_text, type="claim", start_pos=0, end_pos=0)
            premise_unit = ArgumentUnit(uuid=uuid4(), text=premise_text, type="premise", start_pos=0, end_pos=0)
            claim_context, premise_context = self.find_context_for_units(original_text, claim_unit, premise_unit)
            
            user_prompt = f"""Claim with context: {claim_context}
Evidence with context: {premise_context}

Focus on this specific claim: "{claim_text}"
Focus on this specific evidence: "{premise_text}"

Stance:"""
        else:
            user_prompt = f"""Claim: {claim_text}
Evidence: {premise_text}
Stance:"""
            
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (self.system_prompt_stance_classification + ("\n\n" + self.few_shot_stance_examples if use_few_shot else "")).strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                max_tokens=5
            )
            result = response.choices[0].message.content.strip().lower()  # type: ignore
            if any(x in result for x in ("refute", "con")):
                return "con"
            elif any(x in result for x in ("support", "pro")):
                return "pro"
            else:
                logger.warning(f"Unexpected stance output: {result} for Claim: '{claim_text}' | Premise: '{premise_text}'")
                return "unidentified"
        except Exception as e:
            logger.warning(f"❌ Model {model} failed for stance classification: {e}")
            return "unidentified"
            
    def classify_adus(self, text: str, use_few_shot: bool | None = None) -> UnlinkedArgumentUnits:
        sentences = split_into_sentences(text)
        logger.info(f"Found {len(sentences)} sentences in the input text")

        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []

        model_to_use = "gpt-4.1"
        current_pos = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            adu_type = self.classify_sentence_with_context(sentences, i, model_to_use, use_few_shot)

            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence) if start_pos != -1 else -1

            logger.debug(
                f"Sentence: '{sentence}' | Predicted as: {adu_type} | Start: {start_pos} | End: {end_pos}"
            )

            if adu_type not in ("claim", "premise"):
                logger.warning(f"Skipping sentence due to uncertain ADU type: '{sentence}' → {adu_type}")
                if start_pos != -1:
                    current_pos = end_pos
                continue

            adu = ArgumentUnit(
                uuid=uuid4(),
                text=sentence,
                type=adu_type,
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=None
            )

            if adu_type == "claim":
                claims.append(adu)
            else:
                premises.append(adu)

            if start_pos != -1:
                current_pos = end_pos

        return UnlinkedArgumentUnits(claims=claims, premises=premises)
    
    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str, use_few_shot: bool | None = None) -> LinkedArgumentUnitsWithStance:
        result_linked_arguments: List[StanceRelation] = []
        model_to_use = "gpt-4.1"

        for relation in linked_argument_units.claims_premises_relationships:
            claim = next((c for c in linked_argument_units.claims if c.uuid == relation.claim_id), None)
            if claim is None:
                logger.warning(f"No Claim found for this relationship: {relation} --> Continuing loop")
                continue

            logger.debug(f"Processing Claim: {claim.text}")

            if not relation.premise_ids:
                logger.warning(f"No premises found for claim '{claim.text}' ---> Continuing loop")
                continue

            for pid in relation.premise_ids:
                premise = next((p for p in linked_argument_units.premises if p.uuid == pid), None)
                if not premise:
                    logger.warning(f"No premise found for the ID {pid} ---> Continuing loop")
                    continue

                logger.debug(f"  → With Premise: {premise.text}")
                
                stance_relationship = self.classify_stance_single(
                    claim.text, 
                    premise.text, 
                    model_to_use, 
                    original_text=originalText,
                    use_few_shot=use_few_shot
                )
                
                logger.debug(f"Claim: '{claim.text}' | Premise: '{premise.text}' -> Relationship: {stance_relationship}")
                
                result_linked_arguments.append(
                    StanceRelation(
                        claim_id=claim.uuid,
                        premise_id=premise.uuid,
                        stance=stance_relationship,
                        confidence=None
                    )
                )

        return LinkedArgumentUnitsWithStance(
            original_text=originalText,
            claims=linked_argument_units.claims,
            premises=linked_argument_units.premises,
            stance_relations=result_linked_arguments
        )

def test_model():
    """
    Tests the full pipeline for the TinyLlama LLM Classifier.
    This includes classifying ADUs, linking claims to premises, and classifying stance.
    """
    
    # Factory: Instantiate the correct class with its specific parameters
    miner: AduAndStanceClassifier = OpenAILLMClassifier()
    
    claim_linker = OpenAIClaimPremiseLinker()
    example_text = "Climate Change is made up. The measurements of temperature were only recorded the last 100 years, before that there could've been even hotter times. Urban gardening is not just a trend; it is a necessary adaptation to modern urban life. Cities are increasingly crowded, and access to fresh produce is often limited in low-income neighborhoods. By turning rooftops, balconies, and vacant lots into green spaces, residents can take control of their food sources. This not only improves nutrition but also promotes community building and environmental awareness. Moreover, urban gardens help reduce the urban heat island effect, making cities more livable during extreme weather. While some argue that the scale of urban gardening is too small to make a real impact, its cumulative effects—both social and ecological—can be profound."

    # The rest of the pipeline is IDENTICAL for both models because they share the same interface.
    
    # --- Step 1: Classify ADUs to get unlinked claims and premises ---
    logger.info("--- Running Step 1: Classify ADUs using OpenAI ---")
    unlinked_adus = miner.classify_adus(example_text)
    logger.info(f"Found Claims: {len(unlinked_adus.claims)}")
    logger.info(f"Found Premises: {len(unlinked_adus.premises)}")
    logger.info("--------------------")

    # --- Step 2: Link claims to premises using the OpenAI linker ---
    logger.info("--- Running Step 2: Linking Claims to Premises (OpenAI) ---")
    try:
        linked_adus = claim_linker.link_claims_to_premises(unlinked_adus)
        logger.info(f"Successfully linked ADUs.")
        logger.info("--------------------")
    except (openai.AuthenticationError, ValueError) as e:
        logger.error(f"ERROR: Could not run linking step. Please check your OpenAI API key. Details: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during linking: {e}")
        return

    # --- Step 3: Classify the stance for the linked units ---
    logger.info(f"--- Running Step 3: Classify Stance using OpenAI ---")
    final_structure = miner.classify_stance(
        linked_argument_units=linked_adus, originalText=example_text
    )
    logger.info(f"Generated {len(final_structure.stance_relations)} stance relations.")
    logger.info("--------------------")

    # --- Final Output ---
    def _convert_to_json(o):
        if isinstance(o, UUID): return str(o)
        if isinstance(o, dict): return {k: _convert_to_json(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert_to_json(v) for v in o]
        return o

    raw_dict = asdict(final_structure)
    final_json = json.dumps(_convert_to_json(raw_dict), indent=2)

    logger.info("\n--- Final Argument Structure Output ---")
    logger.info(final_json)
