from litellm import ModelResponseStream
from ..stream_lib import Stream

class ContentDeltaStream(Stream[ModelResponseStream, str]):
    """
    A transformer stream that extracts the content delta out of each
    `litellm.ModelResponseStream` chunk.
    """

    def recv(self, data): 
        content = data.choices[0].delta.content
        if isinstance(content, str):
            self.send(content)
        