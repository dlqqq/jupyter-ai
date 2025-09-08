from litellm import acompletion, ModelResponseStream
from ..stream_lib import StreamSource

class LitellmStreamSource(StreamSource[ModelResponseStream]):
    async def start(self, litellm_args: dict):
        response_aiter = await acompletion(
            **litellm_args
        )

        async for chunk in response_aiter:
            self.send(chunk)
