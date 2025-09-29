from abc import ABC, abstractmethod
from typing import Union
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage


class BaseAgent(ABC):

    @abstractmethod
    def run(self, state: dict) -> Union[Command, HumanMessage, AIMessage]:
        pass

    def __call__(self, state: dict) -> Union[Command, HumanMessage, AIMessage]:
        return self.run(state)
