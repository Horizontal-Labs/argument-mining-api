from pathlib import Path
import openai
import torch
import json
import re
from typing import List, Dict
from uuid import UUID as uuid, uuid4
from dataclasses import asdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from peft import PeftModel
import logging

from .openai_claim_premise_linker import OpenAIClaimPremiseLinker

from ..models.argument_units import ArgumentUnit, ClaimPremiseRelationship, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, UnlinkedArgumentUnits
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from app.log import logger


class PeftEncoderModelLoader(AduAndStanceClassifier):
    """
    This class lets you load and use a model like ModernBERT, or any other model, for argument mining tasks.
    It supports ADU identification, ADU classification, and stance classification.
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_paths: Dict[str, str],
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.base_model_path = base_model_path

        root = Path(__file__).parent
        print(f"Root path: {root}")
        self.adapter_paths = {
            task: str((root / rel_path))
            for task, rel_path in adapter_paths.items()
        }
        print(f"Adapter paths: {self.adapter_paths}")
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with trust_remote_code: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Models loaded on demand
        self.models = {}

        # Task configurations
        self.task_configs = {
            "adu_identification": {
                "labels": ["No", "Yes"],
                "task_type": "token_classification",
            },
            "adu_classification": {
                "labels": ["claim", "premise"],
                "task_type": "sequence_classification",
            },
            "stance_classification": {
                "labels": ["con", "pro"],
                "task_type": "sequence_classification",
            },
        }

        logger.info("ModernBERT Argument Miner initialized")

    def load_model_for_task(self, task_name: str):
        if task_name in self.models:
            return self.models[task_name]

        logger.info(f"Loading model for task: {task_name}")

        try:
            if self.task_configs[task_name]["task_type"] == "token_classification":
                base_model = AutoModelForTokenClassification.from_pretrained(
                    self.base_model_path,
                    num_labels=len(self.task_configs[task_name]["labels"]),
                    trust_remote_code=True  # Required for ModernBERT
                )
            else:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model_path,
                    num_labels=len(self.task_configs[task_name]["labels"]),
                    trust_remote_code=True  # Required for ModernBERT
                )
        except ImportError as e:
            logger.error(f"Failed to load ModernBERT model: {e}")
            logger.info("Attempting fallback to BERT-compatible model loading...")
            # Fallback: Try loading without trust_remote_code first, then with generic model classes
            try:
                if self.task_configs[task_name]["task_type"] == "token_classification":
                    from transformers import BertForTokenClassification
                    base_model = BertForTokenClassification.from_pretrained(
                        self.base_model_path,
                        num_labels=len(self.task_configs[task_name]["labels"]),
                        ignore_mismatched_sizes=True
                    )
                else:
                    from transformers import BertForSequenceClassification
                    base_model = BertForSequenceClassification.from_pretrained(
                        self.base_model_path,
                        num_labels=len(self.task_configs[task_name]["labels"]),
                        ignore_mismatched_sizes=True
                    )
                logger.info("Successfully loaded model using BERT fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback loading also failed: {fallback_error}")
                raise e

        model = PeftModel.from_pretrained(base_model, self.adapter_paths[task_name])
        model = model.to(self.device)
        model.eval()

        self.models[task_name] = model
        return model

    def identify_adus(
        self, text: str, confidence_threshold: float = 0.5
    ) -> List[ArgumentUnit]:
        model = self.load_model_for_task("adu_identification")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True,
        ).to(self.device)

        offset_mapping = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

        labels = self.task_configs["adu_identification"]["labels"]
        pred_labels = [labels[p] for p in predictions[0].cpu().numpy()]
        confidences = probabilities[0, :, 1].cpu().numpy()

        adus = []
        current_span = None

        for i, (pred_label, confidence, offset) in enumerate(
            zip(pred_labels, confidences, offset_mapping)
        ):
            if pred_label == "Yes" and confidence > confidence_threshold:
                if current_span is None:
                    current_span = {
                        "start": offset[0].item(),
                        "end": offset[1].item(),
                        "confidence": confidence,
                    }
                else:
                    current_span["end"] = offset[1].item()
                    current_span["confidence"] = max(
                        current_span["confidence"], confidence
                    )
            else:
                if current_span is not None:
                    span_text = text[
                        current_span["start"] : current_span["end"]
                    ].strip()
                    if span_text:
                        adus.append(
                            ArgumentUnit(
                                uuid=uuid4(),
                                text=span_text,
                                start_pos=current_span["start"],
                                end_pos=current_span["end"],
                                type="unknown",
                                confidence=float(current_span["confidence"]),
                            )
                        )
                    current_span = None

        # handle final span
        if current_span is not None:
            span_text = text[current_span["start"] : current_span["end"]].strip()
            if span_text:
                adus.append(
                    ArgumentUnit(
                        uuid=uuid4(),
                        text=span_text,
                        start_pos=current_span["start"],
                        end_pos=current_span["end"],
                        type="unknown",
                        confidence=float(current_span["confidence"]),
                    )
                )

        return adus

    def classify_adu_type(self, adu_text: str) -> tuple[str, float]:
        model = self.load_model_for_task("adu_classification")

        inputs = self.tokenizer(
            adu_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities).item()

        labels = self.task_configs["adu_classification"]["labels"]
        return labels[prediction.item()], confidence

    def _classify_stance_pair(
        self, claim_text: str, premise_text: str
    ) -> tuple[str, float]:
        """Helper method to classify stance for a single claim-premise pair."""
        model = self.load_model_for_task("stance_classification")

        combined_text = f"[CLS] {claim_text} [SEP] {premise_text} [SEP]"
        inputs = self.tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities).item()

        labels = self.task_configs["stance_classification"]["labels"]
        return labels[prediction.item()], confidence

    def extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    # --- INTERFACE METHOD IMPLEMENTATIONS ---

    def classify_adus(
        self, text: str, use_few_shot: bool | None = None, use_sentence_fallback: bool = True
    ) -> UnlinkedArgumentUnits:
        """
        Extracts and labels argumentative units from text.
        This method fulfills the 'classify_adus' contract from the interface.
        """
        logger.info(f"Processing text of length {len(text)}")
        adus = self.identify_adus(text)

        # fallback to sentences
        if not adus and use_sentence_fallback:
            logger.info("No ADUs identified, using sentence-level fallback")
            sentences = self.extract_sentences(text)
            adus = []
            current_pos = 0
            for sentence in sentences:
                start_pos = text.find(sentence, current_pos)
                if start_pos != -1:
                    adus.append(
                        ArgumentUnit(
                            uuid=uuid4(),
                            text=sentence,
                            start_pos=start_pos,
                            end_pos=start_pos + len(sentence),
                            type="unknown",
                            confidence=0.5,  # Default confidence
                        )
                    )
                    current_pos = start_pos + len(sentence)

        # classify ADU types
        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []
        for adu in adus:
            adu_type, conf = self.classify_adu_type(adu.text)
            adu.type = adu_type
            adu.confidence = (
                min(adu.confidence if adu.confidence is not None else conf, conf)
            )
            (claims if adu_type == "claim" else premises).append(adu)

        return UnlinkedArgumentUnits(premises=premises, claims=claims)

    def classify_stance(
        self,
        linked_argument_units: LinkedArgumentUnits,
        originalText: str,
        use_few_shot: bool | None = None,
    ) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance for pre-linked claims and premises.
        This method fulfills the 'classify_stance' contract from the interface.
        """
        stance_relations: List[StanceRelation] = []

        # Create a lookup map for faster access to ADU objects
        claims_map = {c.uuid: c for c in linked_argument_units.claims}
        premises_map = {p.uuid: p for p in linked_argument_units.premises}

        # Iterate through the defined relationships
        for relationship in linked_argument_units.claims_premises_relationships:
            claim_id = relationship.claim_id
            if claim_id is None:
                logger.warning("Encountered a relationship with claim_id=None. Skipping.")
                continue
            claim = claims_map.get(claim_id)

            if not claim:
                logger.warning(f"Claim with ID {claim_id} not found. Skipping.")
                continue

            for premise_id in relationship.premise_ids:
                premise = premises_map.get(premise_id)
                if not premise:
                    logger.warning(
                        f"Premise with ID {premise_id} not found. Skipping."
                    )
                    continue

                # Use the helper method to get stance for the pair
                stance, conf = self._classify_stance_pair(claim.text, premise.text)

                stance_relations.append(
                    StanceRelation(
                        claim_id=claim_id,
                        premise_id=premise_id,
                        stance=stance,
                        confidence=conf,
                    )
                )

        return LinkedArgumentUnitsWithStance(
            original_text=originalText,
            claims=linked_argument_units.claims,
            premises=linked_argument_units.premises,
            stance_relations=stance_relations,
        )

    def process_json_input(self, json_input: str) -> str:
        """Wrapper method to process a raw JSON string input."""
        try:
            input_data = json.loads(json_input)
            text = input_data.get("text", "")
            if not text:
                raise ValueError("No 'text' field found in input JSON")

            # Call the correctly named interface method
            result = self.classify_adus(text)

            # serialize dataclass → dict → JSON, converting UUIDs to strings
            def _convert(o):
                if isinstance(o, uuid):
                    return str(o)
                if isinstance(o, dict):
                    return {k: _convert(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_convert(v) for v in o]
                return o

            raw = asdict(result)
            clean = _convert(raw)
            return json.dumps(clean, indent=2)

        except Exception as e:
            logger.error(f"Error processing JSON input: {str(e)}")
            return json.dumps({"error": str(e), "status": "failed"}, indent=2)
class NonTrainedEncoderModelLoader(AduAndStanceClassifier):
    """
    Implementation of the classifier interface for Non Tuned Encoder models.
    This class does not use PEFT adapters.
    """
    def __init__(self, model_paths: Dict[str, str], device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models and tokenizer from the provided paths
        type_model_path = model_paths['type_model_path']
        stance_model_path = model_paths['stance_model_path']
        
        # Check if paths are HuggingFace model IDs or local paths
        if '/' in type_model_path and not type_model_path.startswith(('/', '.', 'C:', 'D:')):
            # It's a HuggingFace model ID with subfolder
            logger.info(f"Loading DeBERTa models from HuggingFace:")
            logger.info(f"  Type model: {type_model_path}")
            logger.info(f"  Stance model: {stance_model_path}")
            
            # For HuggingFace repos with subfolders, we need to download files from the subfolder
            from huggingface_hub import snapshot_download
            import tempfile
            import os
            
            # Extract repo_id and subfolder
            
            repo_id= "microsoft/deberta-v3-base"
            # Download the models
            cache_dir = os.path.join(tempfile.gettempdir(), "deberta_cache")
            type_model_path = snapshot_download(repo_id= repo_id,cache_dir=cache_dir)
            type_model_path = os.path.join(type_model_path)
            
            stance_model_path = snapshot_download(repo_id= repo_id, cache_dir=cache_dir)
            stance_model_path = os.path.join(stance_model_path)
            
        elif not type_model_path.startswith(('/', '.', 'C:', 'D:')):
            # It's a simple HuggingFace model ID
            logger.info(f"Loading DeBERTa models from HuggingFace:")
            logger.info(f"  Type model: {type_model_path}")
            logger.info(f"  Stance model: {stance_model_path}")
        else:
            # It's a local path
            root = Path(__file__).parent
            type_model_path = str(root / type_model_path)
            stance_model_path = str(root / stance_model_path)
            logger.info(f"Loading DeBERTa models from local paths:")
            logger.info(f"  Type model: {type_model_path}")
            logger.info(f"  Stance model: {stance_model_path}")
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(type_model_path)
        self.type_model = AutoModelForSequenceClassification.from_pretrained(type_model_path).to(self.device)
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_path).to(self.device)
        self.type_model.config.id2label = {
            0: "premise",
            1: "claim",
        }
        self.type_model.config.label2id = {
            "premise": 0,
            "claim":  1,
        }
        self.stance_model.config.id2label = {0: "con", 1: "pro"}
        self.stance_model.config.label2id = {"con": 0, "pro": 1}
        
        self.type_model.eval()
        self.stance_model.eval()
        logger.info("DeBERTa Argument Miner initialized")

    def extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _classify_adu_type(self, sentence: str) -> tuple[str, float]:
        """Helper method to classify a single sentence as a claim or premise."""
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.type_model(**inputs)
            probs = torch.softmax(output.logits, dim=-1)
            pred = torch.argmax(probs).item()
            label = "claim" if pred == 1 else "premise"
            conf = torch.max(probs).item()
            return label, conf

    def _classify_stance_pair(self, claim_text: str, premise_text: str) -> tuple[str, float]:
        """Helper method to classify stance for a single claim-premise pair."""
        combined = f"{claim_text} [SEP] {premise_text}"
        inputs = self.tokenizer(combined, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.stance_model(**inputs)
            probs = torch.softmax(output.logits, dim=-1)
            pred = torch.argmax(probs).item()
            conf = torch.max(probs).item()
        # Assuming the model's labels are {0: 'con', 1: 'pro'}
        label = 'pro' if pred == 1 else 'con'
        return label, conf

    # --- INTERFACE METHOD IMPLEMENTATIONS ---

    def classify_adus(self, text: str, use_few_shot: bool | None = None) -> UnlinkedArgumentUnits:
        """
        Extracts and labels argumentative units from text by treating each sentence as a potential ADU.
        """
        sentences = self.extract_sentences(text)
        claims, premises = [], []
        current_pos = 0

        for sentence in sentences:
            start = text.find(sentence, current_pos)
            if start == -1: continue
            end = start + len(sentence)
            current_pos = end
            
            adu_type, conf = self._classify_adu_type(sentence)
            adu = ArgumentUnit(uuid=uuid4(), text=sentence, start_pos=start, end_pos=end, type=adu_type, confidence=conf)
            (claims if adu_type == 'claim' else premises).append(adu)

        return UnlinkedArgumentUnits(claims=claims, premises=premises)

    def classify_stance(
        self,
        linked_argument_units: LinkedArgumentUnits,
        originalText: str,
        use_few_shot: bool | None = None,
    ) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance for pre-linked claims and premises using the fine-tuned stance model.
        """
        stance_relations: List[StanceRelation] = []
        claims_map = {c.uuid: c for c in linked_argument_units.claims}
        premises_map = {p.uuid: p for p in linked_argument_units.premises}

        for relationship in linked_argument_units.claims_premises_relationships:
            claim = claims_map.get(relationship.claim_id) if relationship.claim_id else None
            if not claim:
                continue

            for premise_id in relationship.premise_ids:
                premise = premises_map.get(premise_id)
                if not premise:
                    continue
                
                stance, conf = self._classify_stance_pair(claim.text, premise.text)
                stance_relations.append(
                    StanceRelation(
                        claim_id=claim.uuid,
                        premise_id=premise.uuid,
                        stance=stance,
                        confidence=conf,
                    )
                )

        return LinkedArgumentUnitsWithStance(
            original_text=originalText,
            claims=linked_argument_units.claims,
            premises=linked_argument_units.premises,
            stance_relations=stance_relations,
        )

MODEL_CONFIGS = {
    "modernbert": {
        "loader_class": PeftEncoderModelLoader,
        "params": {
            "base_model_path": "answerdotai/ModernBERT-base",
            "adapter_paths": {
                "adu_identification":   "argument-mining-modernBert/argument-mining-modernbert-adu_identification/checkpoint-9822",
                "adu_classification":   "argument-mining-modernBert/argument-mining-modernbert-adu_classification/checkpoint-9822",
                "stance_classification":"argument-mining-modernBert/argument-mining-modernbert-stance_classification/checkpoint-4911",
            }
        }
    },
    "deberta": {
        "loader_class": NonTrainedEncoderModelLoader,
        "params": {
            "model_paths": {
                "type_model_path": "microsoft/deberta-v3-base",  # Type classification model
                "stance_model_path": "microsoft/deberta-v3-base"  # Stance classification model
            }
        }
    }
}

def test_model(model_name: str):
    """
    Tests the full pipeline using a factory approach to select the correct model loader.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS.")

    config = MODEL_CONFIGS[model_name]
    LoaderClass = config["loader_class"]
    
    # Factory: Instantiate the correct class with its specific parameters
    miner: AduAndStanceClassifier = LoaderClass(**config["params"])
    
    claim_linker = OpenAIClaimPremiseLinker()
    example_text = "Climate Change is made up. The measurements of temperature were only recorded the last 100 years, before that there could've been even hotter times. Urban gardening is not just a trend; it is a necessary adaptation to modern urban life. Cities are increasingly crowded, and access to fresh produce is often limited in low-income neighborhoods. By turning rooftops, balconies, and vacant lots into green spaces, residents can take control of their food sources. This not only improves nutrition but also promotes community building and environmental awareness. Moreover, urban gardens help reduce the urban heat island effect, making cities more livable during extreme weather. While some argue that the scale of urban gardening is too small to make a real impact, its cumulative effects—both social and ecological—can be profound."

    # The rest of the pipeline is IDENTICAL for both models because they share the same interface.
    
    # --- Step 1: Classify ADUs to get unlinked claims and premises ---
    logger.info(f"--- Running Step 1: Classify ADUs using {model_name} ---")
    unlinked_adus = miner.classify_adus(example_text)
    logger.info(f"Found Claims: {len(unlinked_adus.claims)}")
    logger.info(f"Found Premises: {len(unlinked_adus.premises)}")
    logger.info("--------------------")

    # --- Step 2: Link claims to premises using the OpenAI linker ---
    print("--- Running Step 2: Linking Claims to Premises (OpenAI) ---")
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
    logger.info(f"--- Running Step 3: Classify Stance using {model_name} ---")
    final_structure = miner.classify_stance(
        linked_argument_units=linked_adus, originalText=example_text
    )
    logger.info(f"Generated {len(final_structure.stance_relations)} stance relations.")
    logger.info("--------------------")

    # --- Final Output ---
    def _convert_to_json(o):
        if isinstance(o, uuid): return str(o)
        if isinstance(o, dict): return {k: _convert_to_json(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert_to_json(v) for v in o]
        return o

    raw_dict = asdict(final_structure)
    final_json = json.dumps(_convert_to_json(raw_dict), indent=2)

    logger.info("\n--- Final Argument Structure Output ---")
    logger.info(final_json)
