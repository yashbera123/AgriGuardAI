from pydantic import BaseModel, Field


class AdvisorRequest(BaseModel):
    disease_name: str = Field(..., description="The predicted disease name")
    confidence: float = Field(
        ..., description="Confidence score of the prediction (0-100)"
    )


class AdvisorResponse(BaseModel):
    success: bool = Field(
        ..., description="Whether the advisory was generated successfully"
    )
    recommendation: str = Field(..., description="The AI generated advisory text")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "recommendation": "To treat Tomato Early Blight, ensure proper spacing between plants...",
            }
        }
    }


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's agricultural question")
    history: list = Field(
        default=[], description="List of previous chat messages for context"
    )


class ChatResponse(BaseModel):
    success: bool = Field(
        ..., description="Whether the chat response was generated successfully"
    )
    response: str = Field(..., description="The AI generated response text")
    provider: str = Field(
        ..., description="The AI provider used (e.g., Gemini, Groq, Local)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "response": "Tomatoes generally need 1-2 inches of water per week.",
                "provider": "Gemini",
            }
        }
    }
