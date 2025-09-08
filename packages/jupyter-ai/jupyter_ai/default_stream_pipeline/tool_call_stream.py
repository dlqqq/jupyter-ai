from litellm import ModelResponseStream
from ..litellm_lib import ToolCallList as ToolCallAggregator
from ..stream_lib import Stream

class ToolCallStream(Stream[ModelResponseStream, str]):
    _aggregator: ToolCallAggregator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aggregator = ToolCallAggregator()

    def recv(self, data):
        tool_call_delta = data.choices[0].delta.tool_calls
        self._aggregator += tool_call_delta
        

# - Emit when name & ID are available ('')
# - Emit when tool call is fully received & starts running

# ToolCallEvent


# 'new'
# 'running'
# 'failure'
# 'success'
