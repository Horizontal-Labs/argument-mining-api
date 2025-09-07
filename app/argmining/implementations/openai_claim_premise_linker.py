import logging
import openai
import os
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from uuid import UUID, uuid4

from ..interfaces.claim_premise_linker import ClaimPremiseLinker
from ..models.argument_units import ArgumentUnit, ClaimPremiseRelationship, LinkedArgumentUnits, UnlinkedArgumentUnits
# Handle logger import with fallback to benchmark logging
try:
    from ...log import logger
except ImportError:
    try:
        from app.log import logger
    except ImportError:
        # Fallback to benchmark logging utilities
        try:
            from benchmark.utils.logging_utils import get_logger, setup_logging
            logger = setup_logging(log_level="INFO", progress_bar_compatible=True)
            logger.info("Using benchmark logging utilities as fallback")
        except ImportError:
            # Ultimate fallback - basic logging
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.info("Using basic logging fallback")

# Handle OPENAI_KEY import with fallback to environment
try:
    from ..config import OPENAI_KEY
except ImportError:
    try:
        from argmining.config import OPENAI_KEY
    except ImportError:
        # Fallback to environment variables
        logger.warning("OPENAI_KEY not found in config.py or argmining.config.py, using environment variables")   
        OPENAI_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPEN_AI_KEY")

class ClaimPremiseRelationshipPydanticModel(BaseModel):
    claim_id: UUID | None
    premise_ids : List[UUID] | None

class LinkingOutputPydanticModel(BaseModel):
    claims_premises_relationships: List[ClaimPremiseRelationshipPydanticModel]

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

function_schema = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="link_premises_to_claims",
        description="Link each premise to the claim.",
        parameters=LinkingOutputPydanticModel.model_json_schema()
    )
    
)



class OpenAIClaimPremiseLinker(ClaimPremiseLinker):
    """
    Implementation of ClaimPremiseLinker using OpenAI's API to link claims and premises.
    This class uses the OpenAI API to link premises to claims based on their content.
    """

    def __init__(self):
            self.client = openai.OpenAI(api_key=OPENAI_KEY)

    def link_claims_to_premises(
        self,
        unlinked_argument_units: UnlinkedArgumentUnits,
        max_retries: int = 3
    ) -> LinkedArgumentUnits:
        claims  = unlinked_argument_units.claims
        premises = unlinked_argument_units.premises

        system_prompt = """
    You are an expert in argument mining. Your job is to link PREMISES to the CLAIMS they support or attack.

    Instructions:
    - You will receive a list of CLAIMS and a list of PREMISES.
    - Each premise belongs to at least one claim.
    """
        # build user prompt
        user_prompt = "\n".join(
            ["CLAIMS:"] +
            [f"- claimId: {c.uuid} | {c.text}" for c in claims] +
            ["", "PREMISES:"] +
            [f"- premiseId: {p.uuid} | {p.text}" for p in premises]
        )

        for attempt in range(1, max_retries + 1):
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user",   "content": user_prompt},
                ],
                tools=[function_schema],
                tool_choice="required",
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                logger.error(f"⚠️ No tool calls returned by OpenAI on attempt {attempt}/{max_retries}, retrying...")
                if attempt < max_retries:
                    continue
                else:
                    raise RuntimeError("No tool calls returned by OpenAI after maximum retries.")
            call = tool_calls[0]
            raw = call.function.arguments

            try:
                # validate against our Pydantic schema
                validated = LinkingOutputPydanticModel.model_validate_json(raw)
            except ValidationError as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Validation failed on attempt {attempt}/{max_retries}, retrying...")
                    continue
                else:
                    # no retries left: bubble up the error
                    logger.fatal("❌ Final validation error, giving up:")
                    logger.fatal(e)
                    raise
            else:
                # success: wrap into your return type
                claims_premises_relationships = [
                    ClaimPremiseRelationship(
                        claim_id=rel.claim_id if rel.claim_id else None,
                        premise_ids=[pid for pid in (rel.premise_ids or []) if pid is not None]
                    )
                    for rel in validated.claims_premises_relationships
                ]

                return LinkedArgumentUnits(
                    claims=claims,
                    premises=premises,
                    claims_premises_relationships=claims_premises_relationships
                )

        # Should never get here
        raise RuntimeError("Reached unexpected code path in link_premises_to_claims")
    

    
def test_linker():
    claims = [
       ArgumentUnit(type='claim', uuid=uuid4(), text="Remote work improves employee productivity.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='claim', uuid=uuid4(), text="AI-generated art is not true creativity.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='claim', uuid=uuid4(), text="Government surveillance programs invade personal privacy.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='claim', uuid=uuid4(), text="Electric vehicles are better for the environment.", start_pos=0, end_pos=0, confidence=1.0),
    ]

    premises = [
       ArgumentUnit(type='premise', uuid=uuid4(), text="EVs have zero tailpipe emissions.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='premise', uuid=uuid4(), text="EV battery mining damages ecosystems.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='premise', uuid=uuid4(), text="Most EVs are charged with electricity generated from fossil fuels.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='premise', uuid=uuid4(), text="AI-generated images mimic style but lack original thought.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='premise', uuid=uuid4(), text="People working remotely often report fewer distractions and better focus.", start_pos=0, end_pos=0, confidence=1.0),
       ArgumentUnit(type='premise', uuid=uuid4(), text="3D-printed organs could reduce transplant wait times.", start_pos=0, end_pos=0, confidence=1.0)
]
    # Create UnlinkedArgumentUnits instance
    unlinked_argument_units = UnlinkedArgumentUnits(
        claims=claims,
        premises=premises
    )

    # Create an instance of the OpenAIClaimPremiseLinker
    linker = OpenAIClaimPremiseLinker()

    result = linker.link_claims_to_premises(unlinked_argument_units)

    print("Claims and their linked premises:")
    for relation in result.claims_premises_relationships:
        claim = next((c for c in claims if c.uuid == relation.claim_id), None)
        if claim:
            print(f"Claim: {claim.text} (ID: {claim.uuid})")
            linked_premises = [p.text for p in premises if p.uuid in relation.premise_ids]
            print(f"  Linked Premises: {', '.join(linked_premises)}")
        else:
            print(f"Claim with ID {relation.claim_id} not found.")
