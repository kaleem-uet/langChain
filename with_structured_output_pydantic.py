from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="Return sentiment of the review either negative or positive")
    pros: list[str] = Field(description="Write down all the pros inside a list")
    cons: list[str] = Field(description="Write down all the cons inside a list")
    name: str = Field(description="Write the name of the reviewer")

# Create Azure Chat model
model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.2,
)

# Enable structured output
structured_model = model.with_structured_output(Review)

# Sample review
review_text = """
I’ve been using the Google Pixel 8 Pro for a few weeks now, and it’s honestly one of the smartest phones I’ve ever owned. The AI features like call screening, Magic Eraser, and voice typing are actually useful and not just gimmicks. The display is bright and vibrant, and the 120Hz refresh rate makes scrolling super smooth.

Battery life is solid—I usually end the day with 20% left. The camera, especially in low light and portrait mode, is phenomenal. Google’s photo processing is top-tier, and the natural colors really appeal to me. The phone feels premium, and the matte finish is fingerprint-resistant.

That said, the face unlock doesn’t work well in dim lighting, and there’s still no proper manual mode for the camera, which is disappointing. Also, at $999, it feels a bit overpriced when you consider some missing pro-level controls.

Pros:
Incredible AI-powered features
Outstanding camera quality
Smooth and vibrant display
Great battery life

Cons:
Face unlock struggles in low light

# Lacks manual camera controls
Relatively expensive

Review by Amrita Bhandari
"""

result = structured_model.invoke(review_text)

# Access the structured data
print(result.name)
print(result)