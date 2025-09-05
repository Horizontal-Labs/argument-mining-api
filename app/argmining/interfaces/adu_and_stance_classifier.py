from abc import ABC, abstractmethod
from typing import List

from ..models.argument_units import LinkedArgumentUnits, LinkedArgumentUnitsWithStance, UnlinkedArgumentUnits

class AduAndStanceClassifier(ABC):
    """Interface for ADU extraction and stance classification."""

    @abstractmethod
    def classify_adus(self, text: str, use_few_shot: bool | None = None) -> UnlinkedArgumentUnits:
        """Extracts and labels argumentative units from text.

        Args:
            text (str):  The raw input document.

        Returns:
            UnlinkedArgumentUnits  List of claims and premises extracted from the text.
        """
        pass

    @abstractmethod
    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str, use_few_shot: bool | None = None) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises.

        :param claims: List of claims to be linked.
        :param premises: List of premises to be linked.
        :param originalText: The original text from which the claims and premises were extracted (Nothing is done with it, just passed for context).
        :return: A list of relationships between claims and premises with their stance.
        """
        pass
