from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

I = TypeVar("I")
O = TypeVar("O")

class Stream(ABC, Generic[I, O]):
    last_input: I
    last_output: O
    receiving_nodes: list['Stream'[O, Any]]

    @abstractmethod
    def recv(self, data: I) -> None:
        """
        Method called when data is streamed into this stream.
        """
        pass

    @abstractmethod
    def send(self, data: O) -> None:
        """
        Method called when data is streamed out of this stream.
        """
        pass


class StreamSource(Stream[None, O]):
    def recv(self):
        raise RuntimeError("StreamSource can only emit data and cannot receive data.")


class StreamSink(Stream[I, None]):
    def send(self):
        raise RuntimeError("StreamSink can only receive data and cannot emit data.")

    
class StreamPipeline():
    def __init__():
        pass
    
    async def start():
        pass

    
s = StreamSource[int]()
s.last_input
s.last_output

s = StreamSink[str]()
s.last_input
s.last_output
