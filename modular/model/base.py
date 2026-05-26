"""Abstract base class for model inference engines."""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
import pandas as pd

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class ModelInference(ABC, Generic[InputType, OutputType]):
    """Abstract base class for model inference engines."""

    @abstractmethod
    def infer(self, input_data: InputType) -> OutputType:
        """Run inference on input data.

        Args:
            input_data: Input data for inference

        Returns:
            Inference results
        """
        pass

    @abstractmethod
    def batch_infer(self, batch_data: list[InputType]) -> list[OutputType]:
        """Run inference on a batch of inputs.

        Args:
            batch_data: List of input data

        Returns:
            List of inference results
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for identification."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        pass
