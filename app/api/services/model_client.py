from app.argmining.implementations.encoder_model_loader import MODEL_CONFIGS
from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
from app.argmining.implementations.hf_chat_llm_classifier import HFChatLLMClassifier

from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from app.argmining.models.argument_units import LinkedArgumentUnitsWithStance
from ...log import logger 

_model_instances: dict[str, AduAndStanceClassifier] = {}

def get_adu_classifier(model_name: str) -> AduAndStanceClassifier:
    if model_name not in _model_instances:
        if model_name == "modernbert" or model_name == "deberta":
            model_config = MODEL_CONFIGS.get(model_name)
            if not model_config:
                raise ValueError(f"Model configuration for {model_name} is not defined.")
            LoaderClass = model_config["loader_class"]
            # Factory: Instantiate the correct class with its specific parameters
            miner: AduAndStanceClassifier = LoaderClass(**model_config["params"])
            _model_instances[model_name] = miner
        elif model_name in ["gpt-4.1", "gpt-5", "gpt-5-mini"]:
            # Create OpenAI classifier with specific model
            _model_instances[model_name] = OpenAILLMClassifier(model_name=model_name)
        elif model_name == "openai":
            # Legacy support for generic "openai" - defaults to gpt-4.1
            _model_instances[model_name] = OpenAILLMClassifier(model_name="gpt-4.1")
        elif model_name == "tinyllama-finetuned":
            # Explicit: finetuned adapter enabled
            _model_instances[model_name] = TinyLLamaLLMClassifier(use_adapter=True)
        elif model_name == "tinyllama-base":
            # Explicitly disable adapter (use base chat model only)
            _model_instances[model_name] = TinyLLamaLLMClassifier(use_adapter=False)
        elif model_name == "llama3-3b":
            # Meta Llama 3.2 3B Instruct (chat tuned)
            _model_instances[model_name] = HFChatLLMClassifier(
                base_model_id="meta-llama/Llama-3.2-3B-Instruct",
                name="Llama3.2-3B-Instruct",
            )
        elif model_name == "qwen2.5-1.5b":
            # Qwen 2.5 1.5B Instruct (chat tuned)
            _model_instances[model_name] = HFChatLLMClassifier(
                base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
                name="Qwen2.5-1.5B-Instruct",
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    return _model_instances[model_name]

def serialize_linked_argument_units_with_stance(obj: LinkedArgumentUnitsWithStance) -> dict:
    return {
        "original_text": obj.original_text,
        "claims": [
            {
                "id": str(claim.uuid),
                "text": claim.text,
            } for claim in obj.claims
        ],
        "premises": [
            {
                "id": str(premise.uuid),
                "text": premise.text,
            } for premise in obj.premises
        ],
        "stance_relations": [
            {
                "claim_id": str(relation.claim_id),
                "premise_id": str(relation.premise_id),
                "stance": str(relation.stance)
            } for relation in obj.stance_relations
        ]
    }

def run_argument_mining(
    adu_model_name: str,
    stance_model_name: str,
    text: str,
    use_few_shot_adu: bool | None = None,
    use_few_shot_stance: bool | None = None,
):
    try:
        logger().info("====================== Step1: ADUs classification ======================")
        adu_model = get_adu_classifier(adu_model_name)
        
        unlinked_adus = adu_model.classify_adus(text, use_few_shot_adu)
        
        # Link claims and premises using a separate linker

        linked_adus = OpenAIClaimPremiseLinker().link_claims_to_premises(unlinked_adus)
        logger().debug(f"Linked ADUs are: {linked_adus}")
        logger().info("====================== Step3: Classify Stances ======================")
        # Classify stance using the stance model
        stance_model = get_adu_classifier(stance_model_name)
        result = stance_model.classify_stance(linked_adus, text, use_few_shot_stance)
        
        result_api = serialize_linked_argument_units_with_stance (result)
        return result_api

    except Exception as e:
        raise RuntimeError(f"Pipeline failed for ADU model '{adu_model_name}' and stance model '{stance_model_name}': {str(e)}")
