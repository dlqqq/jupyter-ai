from langchain.pydantic_v1 import BaseModel
from .aliases import MODEL_ID_ALIASES

class ModelId(BaseModel):
    """
    Representation of a model ID. Objects should be constructed via the
    `from_str()` class method. The `provider_id` and `model_name` fields each
    store a component of the model ID. The model ID as a string is accessed by
    calling `str()` on an instance.

    Note: "Model names" have been referred to as "local model IDs". We are
    moving away from the old terminology because it leaves the phrase "model ID"
    ambiguous as to whether it refers to the entire model ID or just the
    component after the provider ID.
    """
    provider_id: str
    model_name: str

    @classmethod
    def from_str(cls, model_id: str) -> "ModelId":
        resolved_model_id = MODEL_ID_ALIASES.get(model_id, None) or model_id

        if ":" not in model_id:
            raise InvalidModelId(resolved_model_id)
        
        provider_id, model_name = resolved_model_id.split(":", 1)
        return ModelId(provider_id=provider_id, model_name=model_name)
        
    def __str__(self):
        return self.provider_id + ":" + self.model_name

class InvalidModelId(Exception):
    def __init__(self, message):            
        super().__init__(f"Invalid model ID: {message}.")
