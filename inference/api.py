from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import torch
from PIL import Image
import io
import base64
import json
import uuid
import time
import asyncio
from contextlib import asynccontextmanager
import logging

from .pipeline import MultimodalInferencePipeline
from ..models import MultimodalLLM
from ..training.config import GenerationConfig

# Global variables for model loading
model_pipeline: Optional[MultimodalInferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading and cleanup."""
    global model_pipeline
    
    # Load model on startup
    try:
        logging.info("Loading multimodal model...")
        
        # Initialize model (in production, load from saved checkpoint)
        model = MultimodalLLM()
        generation_config = GenerationConfig()
        
        model_pipeline = MultimodalInferencePipeline(
            model=model,
            device="auto",
            generation_config=generation_config
        )
        
        logging.info("Model loaded successfully!")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    if model_pipeline:
        model_pipeline.clear_all_conversations()
        logging.info("Model cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Multimodal LLM API",
    description="API for image captioning, visual question answering, and multimodal chat",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class CaptionRequest(BaseModel):
    prompt: str = Field(default="Describe this image in detail.", description="Caption generation prompt")
    max_new_tokens: int = Field(default=256, ge=1, le=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.8, ge=0.1, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling parameter")


class CaptionResponse(BaseModel):
    caption: str
    inference_time: float
    model_info: Dict[str, Any]


class VQARequest(BaseModel):
    question: str = Field(description="Question about the image")
    max_new_tokens: int = Field(default=128, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")


class VQAResponse(BaseModel):
    answer: str
    question: str
    inference_time: float


class ChatRequest(BaseModel):
    message: str = Field(description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for history tracking")
    max_new_tokens: int = Field(default=256, ge=1, le=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    reset_conversation: bool = Field(default=False, description="Reset conversation history")


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    inference_time: float
    turn_count: int
    has_image: bool


class BatchCaptionRequest(BaseModel):
    prompts: List[str] = Field(description="List of prompts for each image")
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    batch_size: int = Field(default=4, ge=1, le=16, description="Processing batch size")


class BatchCaptionResponse(BaseModel):
    captions: List[str]
    total_inference_time: float
    images_processed: int


class ModelInfoResponse(BaseModel):
    model_info: Dict[str, Any]
    status: str
    loaded_at: str


# Helper functions
def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


async def process_uploaded_image(file: UploadFile) -> Image.Image:
    """Process uploaded image file."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return image.convert("RGB")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint."""
    return {
        "message": "Multimodal LLM API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/caption", "/vqa", "/chat", "/batch-caption", 
            "/model-info", "/health", "/docs"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": model_pipeline.device,
        "timestamp": time.time()
    }


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = model_pipeline.get_model_info()
    
    return ModelInfoResponse(
        model_info=model_info,
        status="loaded",
        loaded_at=time.strftime("%Y-%m-%d %H:%M:%S")
    )


@app.post("/caption", response_model=CaptionResponse)
async def caption_image(
    request: CaptionRequest,
    file: UploadFile = File(..., description="Image file to caption")
):
    """Generate a caption for an uploaded image."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process uploaded image
        image = await process_uploaded_image(file)
        
        # Generate caption
        start_time = time.time()
        caption = model_pipeline.caption_image(
            image=image,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        inference_time = time.time() - start_time
        
        return CaptionResponse(
            caption=caption,
            inference_time=inference_time,
            model_info=model_pipeline.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


@app.post("/caption-base64", response_model=CaptionResponse)
async def caption_image_base64(
    image_base64: str = Form(..., description="Base64 encoded image"),
    prompt: str = Form(default="Describe this image in detail."),
    max_new_tokens: int = Form(default=256),
    temperature: float = Form(default=0.7)
):
    """Generate a caption for a base64 encoded image."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image = decode_base64_image(image_base64)
        
        # Generate caption
        start_time = time.time()
        caption = model_pipeline.caption_image(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        inference_time = time.time() - start_time
        
        return CaptionResponse(
            caption=caption,
            inference_time=inference_time,
            model_info=model_pipeline.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


@app.post("/vqa", response_model=VQAResponse)
async def visual_question_answering(
    request: VQARequest,
    file: UploadFile = File(..., description="Image file")
):
    """Answer a question about an uploaded image."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process uploaded image
        image = await process_uploaded_image(file)
        
        # Generate answer
        start_time = time.time()
        answer = model_pipeline.answer_question(
            image=image,
            question=request.question,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        inference_time = time.time() - start_time
        
        return VQAResponse(
            answer=answer,
            question=request.question,
            inference_time=inference_time
        )
        
    except Exception as e:
        logger.error(f"VQA failed: {e}")
        raise HTTPException(status_code=500, detail=f"VQA failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def multimodal_chat(
    request: ChatRequest,
    file: Optional[UploadFile] = File(None, description="Optional image file")
):
    """Multimodal chat with optional image input."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process image if provided
        image = None
        if file is not None:
            image = await process_uploaded_image(file)
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Generate response
        start_time = time.time()
        chat_result = model_pipeline.chat(
            message=request.message,
            image=image,
            conversation_id=conversation_id,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            reset_conversation=request.reset_conversation
        )
        
        return ChatResponse(
            response=chat_result["response"],
            conversation_id=chat_result["conversation_id"],
            inference_time=chat_result["inference_time"],
            turn_count=chat_result["turn_count"],
            has_image=chat_result["has_image"]
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/batch-caption", response_model=BatchCaptionResponse)
async def batch_caption_images(
    request: BatchCaptionRequest,
    files: List[UploadFile] = File(..., description="List of image files")
):
    """Generate captions for multiple images."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) != len(request.prompts):
        raise HTTPException(
            status_code=400, 
            detail="Number of images must match number of prompts"
        )
    
    try:
        # Process all images
        images = []
        for file in files:
            image = await process_uploaded_image(file)
            images.append(image)
        
        # Generate captions
        start_time = time.time()
        captions = model_pipeline.batch_caption(
            images=images,
            prompts=request.prompts,
            batch_size=request.batch_size,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        total_inference_time = time.time() - start_time
        
        return BatchCaptionResponse(
            captions=captions,
            total_inference_time=total_inference_time,
            images_processed=len(images)
        )
        
    except Exception as e:
        logger.error(f"Batch caption failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch caption failed: {str(e)}")


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a given ID."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    history = model_pipeline.get_conversation_history(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "history": history,
        "turn_count": len(history) // 2 if history else 0
    }


@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history for a given ID."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_pipeline.clear_conversation(conversation_id)
    
    return {
        "message": f"Conversation {conversation_id} cleared",
        "conversation_id": conversation_id
    }


@app.delete("/conversations")
async def clear_all_conversations():
    """Clear all conversation histories."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_pipeline.clear_all_conversations()
    
    return {"message": "All conversations cleared"}


@app.post("/benchmark")
async def run_benchmark(background_tasks: BackgroundTasks):
    """Run inference benchmark (background task)."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # This would run a benchmark in the background
    # For now, return a placeholder response
    
    return {
        "message": "Benchmark started",
        "status": "running",
        "note": "Results will be available via logs"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    
    # For development
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )