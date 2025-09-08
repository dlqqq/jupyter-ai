from pydantic import BaseModel
from typing import Union

class ToolCallState:
    state = 'complete'

class ToolCallDeltaStore(BaseModel):
    id: str | None = None
    type: str | None = None
    index: int | None = None
    fn_name: int | None = None
    fn_args: str | None = None
    exception: Exception | None = None
    output: str | None = None

    def resolve(self) -> ToolCallState:
        pass
