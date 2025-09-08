from pydantic import BaseModel
from typing import Any, Literal

class ToolCall(BaseModel):
    """
    An event that describes the status of a tool call made by the LLM.
    """
    status: Literal['new', 'running', 'failure', 'success']

    id: str

    type: str

    index: int

    fn_name: str | None = None
    """
    Set on 'new' and all events after
    """

    fn_args: dict[str, Any] | None = None
    """
    Set on 'running' and all events after
    """

    exception: Exception | None = None
    """
    Set on 'failure'
    """

    output: str | None = None
    """
    Potentially set on 'success'
    """

class PendingToolCall(BaseModel):
    id: str | None
    type: str | None

    @property
    def status(self) -> ToolCallStatus:
        pass


class ToolCallStatus(BaseModel):
    """
    An event that describes the status of a tool call made by the LLM.
    """
    status: Literal['new', 'running', 'failure', 'success']

    id: str

    type: str

    index: int

    fn_name: str | None = None
    """
    Set on 'new' and all events after
    """

    fn_args: dict[str, Any] | None = None
    """
    Set on 'running' and all events after
    """

    exception: Exception | None = None
    """
    Set on 'failure'
    """

    output: str | None = None
    """
    Potentially set on 'success'
    """
