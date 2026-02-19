import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Given an observation dict, return an action dict."""

    def reset(self) -> None:
        """Reset internal state at the start of a new episode."""
        pass
