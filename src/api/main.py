from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import Dict, Optional

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.gpt2_model import PoetryGPT2
from src.utils.poetry_analysis import PoetryAnalyzer

app = FastAPI(title="Poetry Generator API")

class PoemRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 200
    analyze: Optional[bool] = False

class PoemResponse(BaseModel):
    poem: str
    analysis: Optional[Dict] = None

# Initialize models
gpt2_model = PoetryGPT2()
analyzer = PoetryAnalyzer()

@app.post("/generate/", response_model=PoemResponse)
async def generate_poem(request: PoemRequest):
    try:
        # Generate poem
        poem = gpt2_model.generate_poem(request.prompt, request.max_length)
        
        # Analyze if requested
        analysis = None
        if request.analyze:
            analysis = analyzer.get_poem_stats(poem)
        
        return PoemResponse(poem=poem, analysis=analysis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/")
async def analyze_poem(poem: str):
    try:
        analysis = analyzer.get_poem_stats(poem)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
