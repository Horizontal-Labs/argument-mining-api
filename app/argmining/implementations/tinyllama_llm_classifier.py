from dataclasses import asdict
import json
import uuid
import openai
import torch
import re 
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from uuid import UUID, uuid4

from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, ClaimPremiseRelationship, UnlinkedArgumentUnits
from typing import List
from ...log import log
from ..config import HF_TOKEN 

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

class TinyLLamaLLMClassifier (AduAndStanceClassifier): 
    def __init__ (self): 
        self.base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.adapter_path = os.path.join(os.path.dirname(__file__), "TinyLlama-1.1B-Chat-v1.0_finetuned")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, use_auth_token=HF_TOKEN)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            device_map="auto"
        )
        
        # Load PEFT adapter
        try:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            log().info(f"Successfully loaded PEFT adapter from {self.adapter_path}")
        except Exception as e:
            log().warning(f"Failed to load PEFT adapter from {self.adapter_path}: {e}")
            log().warning("Falling back to base model")
            self.model = base_model
            
        # Use the same system prompts as OpenAI classifier for consistency
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
        self.system_prompt_stance_classification = """You are an assistant for argument mining.
                        You are given a claim and evidence (premise) along with their surrounding context.
                        Determine whether the evidence supports ('pro') or refutes ('con') the claim.
                        Use the context to better understand the relationship between claim and premise.
                        Respond only with one word: "pro" or "con".
                        """
    
    def run_prompt(self, prompt, max_new_tokens=150, max_retries=3, retry_delay=1):
        """
        Runs the prompt on the model with a retry mechanism.
        Args:
            prompt (str): The prompt to send to the model.
            max_new_tokens (int): Number of new tokens to generate.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int or float): Delay (in seconds) between retries.
        Returns:
            str: The model's response.
        Raises:
            Exception: If all attempts fail.
        """
        import time
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.3
                    )
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return result[len(prompt):].strip()
            except Exception as e:
                log().warning(f"⚠️ LLM failed on attempt {attempt}/{max_retries}: {e}")
                last_exception = e
                if attempt < max_retries:
                    time.sleep(retry_delay)
        # If we reach here, all attempts failed
        raise Exception(f"Model failed to run the prompt after {max_retries} attempts") from last_exception
    
    def get_context_window(self, sentences: List[str], target_index: int, window_size: int = 2) -> tuple[str, str, str]:
        """
        Get the context window around a target sentence.
        Returns (before_context, target_sentence, after_context)
        """
        start_idx = max(0, target_index - window_size)
        end_idx = min(len(sentences), target_index + window_size + 1)
        
        before_context = " ".join(sentences[start_idx:target_index]) if target_index > 0 else ""
        target_sentence = sentences[target_index]
        after_context = " ".join(sentences[target_index + 1:end_idx]) if target_index < len(sentences) - 1 else ""
        
        return before_context, target_sentence, after_context
        
    def classify_sentence_with_context(self, sentences: List[str], target_index: int) -> str:
        """
        Executes the prompt to classify a single sentence as a 'claim' or 'premise' with context.
        Uses the same prompt structure as OpenAI classifier.
        """
        before_context, target_sentence, after_context = self.get_context_window(sentences, target_index)
        
        user_prompt = f"""Context before: "{before_context}"    
        TARGET sentence: "{target_sentence}"
        Context after: "{after_context}"

        What is the TARGET sentence?"""

        full_prompt = f"{self.system_prompt_adu_classification.strip()}\n\n{user_prompt.strip()}\n\nAnswer:"

        try:
            response = self.run_prompt(full_prompt, max_new_tokens=10)
            result = response.strip().lower()

            if "claim" in result:
                return "claim"
            elif "premise" in result:
                return "premise"
            else:
                log().warning(f"⚠️ Unexpected ADU classification output: {result} - labeling as 'unknown'")
                return "unknown"
        except Exception as e:
            log().warning(f"❌ TinyLlama failed for ADU classification: {e}")
            log().error("Failed ADU classification. Returning 'unknown'.")
            return "unknown"  # Fallback if model fails

    def classify_sentence(self, sentence: str) -> str:
        """
        Legacy method for backward compatibility - calls contextual version with single sentence.
        """
        return self.classify_sentence_with_context([sentence], 0)
        
    def find_context_for_units(self, text: str, claim_unit: ArgumentUnit, premise_unit: ArgumentUnit, window_size: int = 1) -> tuple[str, str]:
        """
        Find contextual sentences around claim and premise units.
        Returns (claim_context, premise_context)
        """
        sentences = split_into_sentences(text)
        
        # Find which sentences contain the claim and premise
        claim_sentence_idx = -1
        premise_sentence_idx = -1
        
        for i, sentence in enumerate(sentences):
            if claim_unit.text.strip() in sentence:
                claim_sentence_idx = i
            if premise_unit.text.strip() in sentence:
                premise_sentence_idx = i
                
        # Get context for claim
        if claim_sentence_idx >= 0:
            start_idx = max(0, claim_sentence_idx - window_size)
            end_idx = min(len(sentences), claim_sentence_idx + window_size + 1)
            claim_context = " ".join(sentences[start_idx:end_idx])
        else:
            claim_context = claim_unit.text
            
        # Get context for premise
        if premise_sentence_idx >= 0:
            start_idx = max(0, premise_sentence_idx - window_size)
            end_idx = min(len(sentences), premise_sentence_idx + window_size + 1)
            premise_context = " ".join(sentences[start_idx:end_idx])
        else:
            premise_context = premise_unit.text
            
        return claim_context, premise_context
        
    def classify_stance_single(self, claim_text: str, premise_text: str, original_text: str = None) -> str:
        """
        Classifies the stance between a single claim and premise with optional context.
        Uses the same prompt structure as OpenAI classifier.
        """
        if original_text:
            # Create dummy ArgumentUnits to use context finder
            claim_unit = ArgumentUnit(uuid=uuid4(), text=claim_text, type="claim", start_pos=0, end_pos=0)
            premise_unit = ArgumentUnit(uuid=uuid4(), text=premise_text, type="premise", start_pos=0, end_pos=0)
            claim_context, premise_context = self.find_context_for_units(original_text, claim_unit, premise_unit)
            
            user_prompt = f"""Claim with context: {claim_context}
Evidence with context: {premise_context}

Focus on this specific claim: "{claim_text}"
Focus on this specific evidence: "{premise_text}"

Stance:"""
        else:
            # Fallback to original method if no context available
            user_prompt = f"""Claim: {claim_text}
Evidence: {premise_text}
Stance:"""
            
        # Combine system and user prompts for TinyLlama
        full_prompt = f"{self.system_prompt_stance_classification.strip()}\n\n{user_prompt.strip()}\n\nAnswer:"
            
        try:
            response = self.run_prompt(full_prompt, max_new_tokens=10)
            result = response.strip().lower()
            
            if any(x in result for x in ("refute", "con")):
                return "con"
            elif any(x in result for x in ("support", "pro")):
                return "pro"
            else:
                log().warning(f"Unexpected stance output: {result} for Claim: '{claim_text}' | Premise: '{premise_text}'")
                return "unidentified"
        except Exception as e:
            log().warning(f"❌ TinyLlama failed for stance classification: {e}")
            return "unidentified" # Fallback if model fails
        
    
    def classify_adus(self, text: str) -> UnlinkedArgumentUnits:
        """
        Extracts and labels argumentative units from text with contextual awareness.
        """
        sentences = split_into_sentences(text)
        log().info(f"Found {len(sentences)} sentences in the input text")

        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []

        current_pos = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            adu_type = self.classify_sentence_with_context(sentences, i)

            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence) if start_pos != -1 else -1

            if start_pos != -1:
                current_pos = end_pos

            if adu_type not in ("claim", "premise"):
                log().warning(f"Skipping sentence due to uncertain ADU type: '{sentence}' → {adu_type}")
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

        return UnlinkedArgumentUnits(claims=claims, premises=premises)


    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises with contextual awareness.
        Uses the same approach as OpenAI classifier for consistency.

        :param linked_argument_units: The structured links between claims and premises.
        :param originalText: The original text from which the claims and premises were extracted (for context only).
        :return: LinkedArgumentUnitsWithStance object representing the final stance graph.
        """
        result_linked_arguments: List[StanceRelation] = []

        for relation in linked_argument_units.claims_premises_relationships:
            # Find the claim object
            claim = next((c for c in linked_argument_units.claims if c.uuid == relation.claim_id), None)
            if claim is None:
                log().warning(f"No Claim found for this relationship: {relation} --> Continuing loop")
                continue

            log().debug(f"Processing Claim: {claim.text}")

            if not relation.premise_ids:
                log().warning(f"No premises found for claim '{claim.text}' ---> Continuing loop")
                continue

            for pid in relation.premise_ids:
                premise = next((p for p in linked_argument_units.premises if p.uuid == pid), None)
                if not premise:
                    log().warning(f"No premise found for the ID {pid} ---> Continuing loop")
                    continue

                log().debug(f"  → With Premise: {premise.text}")
                
                # Use contextual stance classification - same as OpenAI version
                stance_relationship = self.classify_stance_single(
                    claim.text, 
                    premise.text, 
                    original_text=originalText  # Pass original text for context
                )
                
                log().debug(f"Claim: '{claim.text}' | Premise: '{premise.text}' -> Relationship: {stance_relationship}")
                
                result_linked_arguments.append(
                    StanceRelation(
                        claim_id=claim.uuid,
                        premise_id=premise.uuid,
                        stance=stance_relationship,
                        confidence=None # TinyLlama doesn't directly provide confidence score
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
    miner: AduAndStanceClassifier = TinyLLamaLLMClassifier()
    
    claim_linker = OpenAIClaimPremiseLinker()
    example_text = "Climate Change is made up. The measurements of temperature were only recorded the last 100 years, before that there could've been even hotter times. Urban gardening is not just a trend; it is a necessary adaptation to modern urban life. Cities are increasingly crowded, and access to fresh produce is often limited in low-income neighborhoods. By turning rooftops, balconies, and vacant lots into green spaces, residents can take control of their food sources. This not only improves nutrition but also promotes community building and environmental awareness. Moreover, urban gardens help reduce the urban heat island effect, making cities more livable during extreme weather. While some argue that the scale of urban gardening is too small to make a real impact, its cumulative effects—both social and ecological—can be profound."

    # The rest of the pipeline is IDENTICAL for both models because they share the same interface.
    
    # --- Step 1: Classify ADUs to get unlinked claims and premises ---
    log().info(f"--- Running Step 1: Context-Aware ADU Classification using TinyLlama ---")
    unlinked_adus = miner.classify_adus(example_text)
    log().info(f"Found Claims: {len(unlinked_adus.claims)}")
    log().info(f"Found Premises: {len(unlinked_adus.premises)}")
    log().info("--------------------")

    # --- Step 2: Link claims to premises using the OpenAI linker ---
    log().info("--- Running Step 2: Linking Claims to Premises (OpenAI) ---")
    try:
        linked_adus = claim_linker.link_claims_to_premises(unlinked_adus)
        log().info(f"Successfully linked ADUs.")
        log().info("--------------------")
    except (openai.AuthenticationError, ValueError) as e:
        log().error(f"ERROR: Could not run linking step. Please check your OpenAI API key. Details: {e}")
        return
    except Exception as e:
        log().error(f"An unexpected error occurred during linking: {e}")
        return

    # --- Step 3: Classify the stance for the linked units ---
    log().info(f"--- Running Step 3: Context-Aware Stance Classification using TinyLlama ---")
    final_structure = miner.classify_stance(
        linked_argument_units=linked_adus, originalText=example_text
    )
    log().info(f"Generated {len(final_structure.stance_relations)} stance relations.")
    log().info("--------------------")

    # --- Final Output ---
    def _convert_to_json(o):
        if isinstance(o, UUID): return str(o)
        if isinstance(o, dict): return {k: _convert_to_json(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert_to_json(v) for v in o]
        return o

    raw_dict = asdict(final_structure)
    final_json = json.dumps(_convert_to_json(raw_dict), indent=2)

    log().info("\n--- Final Argument Structure Output ---")
    log().info(final_json)