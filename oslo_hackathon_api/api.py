"""
Oslo AI Hackathon API
=====================
FastAPI backend providing chat endpoints for different AI models and tabular prediction capabilities.

**Features:**
- Chat endpoints for OpenAI and Cohere models
- RPT-1 (Retrieval-based Prediction for Tables) model for tabular data predictions
- Streaming chat responses
- File upload support for predictions

Run with: uvicorn api:app --reload
Access Swagger UI at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uvicorn
import json
import asyncio
import os
import requests
from fastapi import UploadFile, File
import tempfile
from pathlib import Path

# Import the chat agents
from agent import chat_with_openai, chat_with_cohere, get_or_refresh_token, stream_chat_with_openai

# Import RPT prediction functions
from rpt import predict_with_rpt1, predict_from_file

# Import German, Swedish, UK, and Finland warehouse/supplier/health agents
import german_agent
import sweden_agent
import uk_agent
import agent_finland

# Import German warehouse agent
from german_agent import final_result


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Oslo AI Hackathon API",
    description="""
    **DEIG AI Hackathon API**
    
    This API provides comprehensive AI capabilities including chat interfaces and tabular data predictions.
    
    **Chat Models:**
    
    *OpenAI Models:*
    - gpt-5
    - gpt-4o
    - gpt-35-turbo
    
    *Cohere Models:*
    - cohere--command-a-reasoning (with optional thinking mode)
    
    **Prediction Models:**
    
    *SAP RPT-1 (Retrieval-based Prediction for Tables):*
    - Small model: Fast predictions for smaller datasets
    - Large model: Enhanced accuracy for complex patterns
    - Supports classification and regression tasks
    - Direct JSON input or CSV/JSON file upload
    
    **Key Features:**
    - Real-time streaming chat responses
    - Tabular data predictions with context-aware AI
    - File upload support (CSV/JSON)
    - Auto-detection of prediction targets
    - Interactive API documentation
    """,
    version="1.0.0",
    contact={
        "name": "DEIG AI Hackathon Team",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(
        ...,
        description="The message to send to the AI model",
        example="Hello! Can you explain quantum computing in simple terms?"
    )
    model: Optional[str] = Field(
        default="gpt-5",
        description="AI model to use (gpt-5, gpt-4o, gpt-35-turbo, etc.)",
        example="gpt-5"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="Your message")
    response: str = Field(..., description="The AI model's response")
    model_used: str = Field(..., description="AI model that was used")
    timestamp: str = Field(..., description="Timestamp of the response")


class CohereRequest(BaseModel):
    """Request model for Cohere chat endpoint."""
    message: str = Field(
        ...,
        description="The message to send to the Cohere model",
        example="Explain the theory of relativity in simple terms."
    )
    model: Optional[str] = Field(
        default="cohere--command-a-reasoning",
        description="Cohere model to use",
        example="cohere--command-a-reasoning"
    )
    enable_thinking: Optional[bool] = Field(
        default=False,
        description="Enable thinking mode for more detailed reasoning (slower response)",
        example=False
    )
    frequency_penalty: Optional[float] = Field(
        default=0.8,
        description="Frequency penalty for response generation (0.0 to 2.0)",
        ge=0.0,
        le=2.0,
        example=0.8
    )


class CohereResponse(BaseModel):
    """Response model for Cohere chat endpoint."""
    message: str = Field(..., description="Your message")
    response: str = Field(..., description="The Cohere model's response")
    thinking: Optional[str] = Field(None, description="The model's reasoning process (if enabled)")
    model_used: str = Field(..., description="Cohere model that was used")
    finish_reason: str = Field(..., description="Reason for completion")
    usage: Dict[str, Any] = Field(..., description="Token usage information")
    timestamp: str = Field(..., description="Timestamp of the response")


class GermanWarehouseRequest(BaseModel):
    """Request model for German warehouse chat endpoint."""
    question: str = Field(
        ...,
        description="Question about warehouse data in English or German",
        example="Wie viele Artikel haben wir im Lager?"
    )
    model: Optional[str] = Field(
        default="gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable streaming mode for Server-Sent Events (SSE)",
        example=False
    )


class GermanWarehouseResponse(BaseModel):
    """Response model for German warehouse chat endpoint."""
    question: str = Field(..., description="Your question")
    response: str = Field(..., description="The AI model's response about warehouse data")
    sql_query: Optional[str] = Field(None, description="The SQL query that was executed")
    row_count: Optional[int] = Field(None, description="Number of rows returned from the database")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Query results as list of dictionaries")
    model_used: str = Field(..., description="AI model that was used")
    timestamp: str = Field(..., description="Timestamp of the response")
    
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning


class UKDemandRequest(BaseModel):
    """Request model for UK demand tracking chat endpoint."""
    question: str = Field(
        ...,
        description="Question about UK supply network demand and delivery tracking",
        example="What are the current demands by theatre?"
    )
    model: Optional[str] = Field(
        default="gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable streaming mode for Server-Sent Events (SSE)",
        example=False
    )


class UKDemandResponse(BaseModel):
    """Response model for UK demand tracking chat endpoint."""
    question: str = Field(..., description="Your question")
    response: str = Field(..., description="The AI model's response about UK demand data")
    sql_query: Optional[str] = Field(None, description="The SQL query that was executed")
    row_count: Optional[int] = Field(None, description="Number of rows returned from the database")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Query results as list of dictionaries")
    model_used: str = Field(..., description="AI model that was used")
    timestamp: str = Field(..., description="Timestamp of the response")
    
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str


class RPTTargetColumn(BaseModel):
    """Model for RPT target column configuration."""
    name: str = Field(..., description="Column name to predict")
    prediction_placeholder: str = Field(
        default="[PREDICT]",
        description="Placeholder value indicating values to predict"
    )
    task_type: str = Field(
        ...,
        description="Type of prediction task (classification or regression)",
        example="classification"
    )


class RPTPredictRequest(BaseModel):
    """Request model for RPT prediction endpoint."""
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of data rows as dictionaries. Use '[PREDICT]' for values to predict.",
        example=[
            {"ID": "35", "PRODUCT": "Couch", "PRICE": "[PREDICT]", "COSTCENTER": "Living Room"},
            {"ID": "44", "PRODUCT": "Office Chair", "PRICE": 150.8, "COSTCENTER": "Office Furniture"},
            {"ID": "27", "PRODUCT": "Sofa", "PRICE": 320.3, "COSTCENTER": "Living Room"},
            {"ID": "38", "PRODUCT": "Table", "PRICE": 129.6, "COSTCENTER": "Office Furniture"},
            {"ID": "56", "PRODUCT": "Lamp", "PRICE": 23.7, "COSTCENTER": "Office Furniture"}
        ]
    )
    index_column: Optional[str] = Field(
        None,
        description="Column name to use as row identifier",
        example="ID"
    )
    target_columns: Optional[List[RPTTargetColumn]] = Field(
        None,
        description="Target column configurations. If not provided, auto-detects from '[PREDICT]' placeholders.",
        example=[
            {
                "name": "PRICE",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "regression"
            }
        ]
    )
    data_schema: Optional[Dict[str, Dict[str, str]]] = Field(
        None,
        description="Schema defining column data types",
        example={"ID": {"dtype": "string"}, "COSTCENTER": {"dtype": "string"},
                 "PRODUCT": {"dtype": "string"}, "PRICE": {"dtype": "numeric"}}
    )
    model_size: str = Field(
        default="small",
        description="RPT model size to use (small or large)",
        example="small"
    )


class RPTPredictResponse(BaseModel):
    """Response model for RPT prediction endpoint."""
    predictions: Dict[str, Any] = Field(..., description="Prediction results from RPT model")
    model_size: str = Field(..., description="Model size used for prediction")
    timestamp: str = Field(..., description="Timestamp of the response")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    "/",
    tags=["General"],
    summary="API Root",
    description="Welcome message and basic API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Oslo AI Hackathon API - Chat & Prediction Services",
        "version": "1.0.0",
        "description": "AI-powered chat and tabular data prediction API",
        "docs": "/docs",
        "health": "/health",
        "capabilities": {
            "chat_models": ["gpt-5", "gpt-4o", "gpt-35-turbo", "cohere--command-a-reasoning"],
            "prediction_models": ["sap-rpt-1-small", "sap-rpt-1-large"],
            "features": ["streaming_chat", "file_upload", "tabular_predictions"]
        },
        "endpoints": {
            "chat": "/api/chat",
            "chat_stream": "/api/chat/stream",
            "cohere_chat": "/api/cohere/chat",
            "cohere_stream": "/api/cohere/stream",
            "german_warehouse": "/api/german/warehouse",
            "german_warehouse_stream": "/api/german/warehouse/stream",
            "swedish_suppliers": "/api/sweden/suppliers",
            "swedish_suppliers_stream": "/api/sweden/suppliers/stream",
            "uk_demand": "/api/uk/demand",
            "uk_demand_stream": "/api/uk/demand/stream",
            "rpt_predict": "/api/rpt/predict",
            "rpt_predict_file": "/api/rpt/predict/file"
        }
    }


@app.get(
    "/health",
    tags=["General"],
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and responsive"
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post(
    "/api/chat",
    tags=["Chat"],
    response_model=ChatResponse,
    summary="Chat with AI",
    description="""
    Send a message to an AI model of your choice and get a response.
    
    **Available Models:**
    - gpt-5
    - gpt-4o
    - gpt-35-turbo
    - And more...
    
    **Example Request:**
    ```json
    {
        "message": "What is the meaning of life?",
        "model": "gpt-5"
    }
    ```
    """
)
async def chat(request: ChatRequest):
    """
    Chat with an AI model.
    
    Send a message and get a response from the specified AI model.
    """
    try:
        # Get response from AI model
        ai_response = chat_with_openai(
            message=request.message,
            model=request.model
        )
        
        return {
            "message": request.message,
            "response": ai_response,
            "model_used": request.model,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.post(
    "/api/chat/stream",
    tags=["Chat"],
    summary="Stream Chat with OpenAI",
    description="""
    Stream responses from OpenAI models (GPT-5, GPT-4o, GPT-35-turbo) using Server-Sent Events (SSE).
    
    This endpoint provides real-time streaming of the AI's response as it's generated.
    
    **Query Parameters:**
    - `message`: The message to send to the AI (required)
    - `model`: The OpenAI model to use (default: gpt-5)
    
    **Event Types:**
    - `status`: Connection and progress updates
    - `text`: Individual response text chunks (word-by-word streaming)
    - `complete`: Final response with full text
    - `error`: Error information
    
    **Client Usage Example (JavaScript):**
    ```javascript
    const eventSource = new EventSource(
        '/api/chat/stream?message=Hello&model=gpt-5'
    );
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'text') {
            document.getElementById('response').innerHTML += data.content;
        } else if (data.type === 'complete') {
            console.log('Full response:', data.full_response);
            eventSource.close();
        }
    };
    ```
    
    **Client Usage Example (Python):**
    ```python
    import requests
    import json
    
    response = requests.post(
        'http://localhost:8000/api/chat/stream',
        params={'message': 'Hello', 'model': 'gpt-5'},
        stream=True
    )
    
    for line in response.iter_lines():
        if line and line.startswith(b'data: '):
            event = json.loads(line[6:])
            
            if event['type'] == 'text':
                print(event['content'], end='', flush=True)
            elif event['type'] == 'complete':
                print('\\n\\nDone!')
    ```
    """
)
async def stream_chat(
    message: str,
    model: str = "gpt-5"
):
    """
    Stream responses from OpenAI models using Server-Sent Events.
    """
    async def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to OpenAI...', 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Send processing status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Accumulate the full response
            full_response = ""
            
            # Stream the response from OpenAI
            for chunk in stream_chat_with_openai(message, model):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk, 'timestamp': datetime.now().isoformat()})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            
            # Send completion event with full response
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Response complete', 'full_response': full_response, 'model_used': model, 'timestamp': datetime.now().isoformat()})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post(
    "/api/cohere/chat",
    tags=["Cohere"],
    response_model=CohereResponse,
    summary="Chat with Cohere",
    description="""
    Send a message to Cohere's command-a-reasoning model and get a response.
    
    **Features:**
    - Advanced reasoning capabilities
    - Optional thinking mode for detailed reasoning process
    - Configurable frequency penalty
    
    **Example Request:**
    ```json
    {
        "message": "Explain quantum computing in simple terms",
        "model": "cohere--command-a-reasoning",
        "enable_thinking": false,
        "frequency_penalty": 0.8
    }
    ```
    
    **Note:** Enabling thinking mode provides insight into the model's reasoning 
    process but increases response time significantly (~15 seconds vs ~3 seconds).
    """
)
async def cohere_chat(request: CohereRequest):
    """
    Chat with Cohere's command-a-reasoning model.
    
    Get sophisticated responses with optional thinking mode that reveals
    the model's reasoning process.
    """
    try:
        # Get response from Cohere model
        result = chat_with_cohere(
            message=request.message,
            model=request.model,
            enable_thinking=request.enable_thinking,
            frequency_penalty=request.frequency_penalty
        )
        
        return {
            "message": request.message,
            "response": result["response"],
            "thinking": result["thinking"],
            "model_used": request.model,
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.post(
    "/api/cohere/stream",
    tags=["Cohere"],
    summary="Stream Chat with Cohere",
    description="""
    Stream responses from Cohere's command-a-reasoning model using Server-Sent Events (SSE).
    
    This endpoint provides real-time streaming of the AI's response as it's generated,
    with parsed text chunks (not raw JSON).
    
    **Query Parameters:**
    - `message`: The message to send to the AI (required)
    - `enable_thinking`: Enable thinking mode (default: false)
    - `frequency_penalty`: Frequency penalty 0.0-2.0 (default: 0.8)
    
    **Event Types:**
    - `status`: Connection and progress updates
    - `thinking_start`: Thinking mode has started (if enabled)
    - `thinking`: Individual thinking text chunks (if enabled)
    - `thinking_end`: Thinking complete (if enabled)
    - `text_start`: Text response has started
    - `text`: Individual response text chunks (stream word-by-word)
    - `complete`: Final response with full text, metadata, and usage
    - `error`: Error information
    
    **Client Usage Example (JavaScript):**
    ```javascript
    const eventSource = new EventSource(
        '/api/cohere/stream?message=Hello&enable_thinking=false'
    );
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
            case 'text':
                // Append text chunks to display
                document.getElementById('response').innerHTML += data.content;
                break;
            case 'complete':
                console.log('Full response:', data.full_response);
                console.log('Usage:', data.usage);
                eventSource.close();
                break;
        }
    };
    ```
    
    **Client Usage Example (Python):**
    ```python
    import requests
    import json
    
    response = requests.post(
        'http://localhost:8000/api/cohere/stream',
        params={'message': 'Hello', 'enable_thinking': False},
        stream=True
    )
    
    for line in response.iter_lines():
        if line and line.startswith(b'data: '):
            event = json.loads(line[6:])
            
            if event['type'] == 'text':
                print(event['content'], end='', flush=True)
            elif event['type'] == 'complete':
                print('\\n\\nUsage:', event['usage'])
    ```
    """
)
async def cohere_stream(
    message: str,
    enable_thinking: bool = False,
    frequency_penalty: float = 0.8
):
    """
    Stream responses from Cohere model using Server-Sent Events.
    """
    async def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to Cohere...', 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Get access token
            access_token = get_or_refresh_token()
            
            # Get environment variables
            BASE_URL = os.getenv("AICORE_BASE_URL")
            RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP")
            
            # Construct the deployment URL
            DEPLOYMENT_URL = f"{BASE_URL}/inference/deployments/d1cd5340a145e7a9"
            
            # Send processing status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Chat request payload
            chat_payload = {
                "model": "cohere--command-a-reasoning",
                "stream": True,  # Enable streaming
                "frequency_penalty": frequency_penalty,
                "thinking": {
                    "type": "enabled" if enable_thinking else "disabled"
                },
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            }
            
            # Make the streaming chat request
            thinking_text = ""
            response_text = ""
            usage_info = None
            finish_reason = None
            
            with requests.post(
                f"{DEPLOYMENT_URL}/chat",
                headers={
                    'AI-Resource-Group': RESOURCE_GROUP,
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {access_token}'
                },
                json=chat_payload,
                stream=True
            ) as response:
                
                if response.status_code != 200:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'API error: {response.status_code}', 'timestamp': datetime.now().isoformat()})}\n\n"
                    return
                
                # Track if we've started streaming thinking or text
                in_thinking = False
                in_text = False
                
                # Stream and parse the response chunks
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            chunk_str = line_str[6:]  # Remove 'data: ' prefix
                            
                            # Skip the [DONE] marker
                            if chunk_str == '[DONE]':
                                continue
                            
                            try:
                                chunk = json.loads(chunk_str)
                                chunk_type = chunk.get('type')
                                
                                # Handle content-start events
                                if chunk_type == 'content-start':
                                    content_type = chunk.get('delta', {}).get('message', {}).get('content', {}).get('type')
                                    if content_type == 'thinking' and enable_thinking:
                                        in_thinking = True
                                        yield f"data: {json.dumps({'type': 'thinking_start', 'message': 'Model is thinking...', 'timestamp': datetime.now().isoformat()})}\n\n"
                                    elif content_type == 'text':
                                        in_text = True
                                        yield f"data: {json.dumps({'type': 'text_start', 'message': 'Streaming response...', 'timestamp': datetime.now().isoformat()})}\n\n"
                                
                                # Handle content-delta events (actual text/thinking chunks)
                                elif chunk_type == 'content-delta':
                                    delta_content = chunk.get('delta', {}).get('message', {}).get('content', {})
                                    
                                    if 'thinking' in delta_content and in_thinking:
                                        thinking_chunk = delta_content['thinking']
                                        thinking_text += thinking_chunk
                                        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_chunk, 'timestamp': datetime.now().isoformat()})}\n\n"
                                    
                                    elif 'text' in delta_content and in_text:
                                        text_chunk = delta_content['text']
                                        response_text += text_chunk
                                        yield f"data: {json.dumps({'type': 'text', 'content': text_chunk, 'timestamp': datetime.now().isoformat()})}\n\n"
                                    
                                    await asyncio.sleep(0.01)  # Small delay for smooth streaming
                                
                                # Handle content-end events
                                elif chunk_type == 'content-end':
                                    if in_thinking:
                                        in_thinking = False
                                        yield f"data: {json.dumps({'type': 'thinking_end', 'message': 'Thinking complete', 'timestamp': datetime.now().isoformat()})}\n\n"
                                    elif in_text:
                                        in_text = False
                                
                                # Handle message-end (contains usage and finish reason)
                                elif chunk_type == 'message-end':
                                    delta = chunk.get('delta', {})
                                    finish_reason = delta.get('finish_reason')
                                    usage_info = delta.get('usage')
                            
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
                
                # Send completion event with metadata
                completion_data = {
                    'type': 'complete',
                    'message': 'Response complete',
                    'full_response': response_text,
                    'finish_reason': finish_reason,
                    'usage': usage_info,
                    'timestamp': datetime.now().isoformat()
                }
                
                if enable_thinking and thinking_text:
                    completion_data['thinking'] = thinking_text
                
                yield f"data: {json.dumps(completion_data)}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# RPT Prediction Endpoints
# ============================================================================

@app.post(
    "/api/rpt/predict",
    tags=["RPT Predictions"],
    response_model=RPTPredictResponse,
    summary="Make predictions with RPT-1 Model",
    description="""
    Use the SAP-RPT-1 Model to make predictions on tabular data.
    
    **How it works:**
    1. Provide your data as a list of row dictionaries
    2. Use `[PREDICT]` as the value for fields you want to predict
    3. The model uses context rows (complete data) to predict missing values
    
    **Example Request:**
    ```json
    {
        "rows": [
            {"ID": "35", "PRODUCT": "Couch", "PRICE": 999.99, "COSTCENTER": "[PREDICT]"},
            {"ID": "44", "PRODUCT": "Office Chair", "PRICE": 150.8, "COSTCENTER": "Office Furniture"},
            {"ID": "27", "PRODUCT": "Sofa", "PRICE": 320.3, "COSTCENTER": "Living Room"}
        ],
        "index_column": "ID",
        "target_columns": [
            {
                "name": "COSTCENTER",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "classification"
            }
        ],
        "model_size": "small"
    }
    ```
    
    **Task Types:**
    - `classification`: For categorical predictions (e.g., category, status)
    - `regression`: For numerical predictions (e.g., price, quantity)
    
    **Model Sizes:**
    - `small`: Faster, suitable for smaller datasets
    - `large`: More accurate, better for complex patterns
    """
)
async def rpt_predict(request: RPTPredictRequest):
    """
    Make predictions using SAP-RPT-1 Model.
    
    Processes tabular data and predicts missing values using the RPT-1 model.
    """
    try:
        # Convert target_columns from Pydantic models to dicts if provided
        target_columns = None
        if request.target_columns:
            target_columns = [col.dict() for col in request.target_columns]
        
        # Make prediction
        result = predict_with_rpt1(
            rows=request.rows,
            index_column=request.index_column,
            target_columns=target_columns,
            data_schema=request.data_schema,
            model_size=request.model_size
        )
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="RPT prediction failed. Check server logs for details."
            )
        
        return {
            "predictions": result,
            "model_size": request.model_size,
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


@app.post(
    "/api/rpt/predict/file",
    tags=["RPT Predictions"],
    summary="Make predictions from CSV/JSON file",
    description="""
    Upload a CSV or JSON file and get predictions using the SAP-RPT-1 Model.
    
    **Supported File Formats:**
    - CSV files (.csv)
    - JSON files (.json)
    
    **JSON Format Options:**
    1. Array of objects:
    ```json
    [
        {"ID": "1", "PRODUCT": "Couch", "PRICE": "[PREDICT]"},
        {"ID": "2", "PRODUCT": "Chair", "PRICE": 150.8}
    ]
    ```
    
    2. Object with 'rows' or 'data' key:
    ```json
    {
        "rows": [
            {"ID": "1", "PRODUCT": "Couch", "PRICE": "[PREDICT]"},
            {"ID": "2", "PRODUCT": "Chair", "PRICE": 150.8}
        ]
    }
    ```
    
    **Query Parameters:**
    - `model_size`: Model size to use (small or large, default: small)
    - `index_column`: Column name to use as row identifier (optional)
    - `task_type`: Prediction task type (classification or regression, default: classification)
    
    **Example cURL:**
    ```bash
    curl -X POST "http://localhost:8000/api/rpt/predict/file?model_size=small&index_column=ID" \
      -F "file=@data.csv"
    ```
    """
)
async def rpt_predict_file(
    file: UploadFile = File(..., description="CSV or JSON file containing the data"),
    model_size: str = "small",
    index_column: Optional[str] = None,
    task_type: str = "classification"
):
    """
    Make predictions from an uploaded CSV or JSON file.
    
    Accepts file uploads and processes them using the RPT-1 model.
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.csv', '.json']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Use .csv or .json files."
            )
        
        # Create temporary file to save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Read and write uploaded file to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Make prediction from file
            result = predict_from_file(
                file_path=temp_file_path,
                index_column=index_column,
                model_size=model_size
            )
            
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="RPT prediction failed. Check server logs for details."
                )
            
            return {
                "predictions": result,
                "model_size": model_size,
                "file_name": file.filename,
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


@app.get(
    "/api/german/warehouse/stream",
    tags=["German Warehouse"],
    summary="German Warehouse Intelligence (Streaming)",
    description="""
    Query German warehouse data using natural language with Server-Sent Events streaming.
    
    This endpoint provides AI-powered access to the German military logistics warehouse database
    with real-time streaming responses.
    
    **Available Data:**
    - WAREHOUSE: Master data with article descriptions, numbers, and monthly forecasts
    - WAREHOUSE_Q424, Q125, Q225, Q325: Quarterly warehouse data
    
    **Example Questions:**
    - "Wie viele Artikel haben wir im Lager?" (How many articles do we have in the warehouse?)
    - "Welche Artikel haben die höchste monatliche Prognose?"
    - "Show me articles with monthly forecast over 1000"
    
    **Example URL:**
    ```
    GET /api/german/warehouse/stream?question=Wie viele Artikel haben wir im Lager?&model=gpt-4o
    ```
    
    **Response Format:**
    Server-Sent Events (SSE) stream with:
    - Text chunks as they're generated
    - Final JSON payload with complete results
    """
)
async def german_warehouse_stream(
    question: str = Query(
        ...,
        description="Question about warehouse data in English or German",
        example="Wie viele Artikel haben wir im Lager?"
    ),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Stream chat responses with German warehouse database.
    """
    async def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Querying warehouse database...', 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Call the german_agent with streaming enabled
            result = final_result(
                prompt=question,
                model_name=model,
                stream=True
            )
            
            # Check for errors
            if result.get("error"):
                yield f"data: {json.dumps({'type': 'error', 'message': result.get('response', 'Unknown error'), 'timestamp': datetime.now().isoformat()})}\n\n"
                return
            
            # Send processing status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Accumulate the full response
            full_response = ""
            
            # Stream the response chunks
            for chunk in result["stream"]:
                chunk_dict = chunk.to_dict()
                choices = chunk_dict.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'type': 'text', 'content': content, 'timestamp': datetime.now().isoformat()})}\n\n"
                        await asyncio.sleep(0.01)
            
            # Send completion event with full JSON response
            final_payload = {
                "type": "complete",
                "message": "Response complete",
                "data": {
                    "question": question,
                    "response": full_response,
                    "sql_query": result.get("sql_query"),
                    "row_count": result.get("row_count"),
                    "results": result.get("results", []),
                    "model_used": model,
                    "timestamp": datetime.now().isoformat()
                }
            }
            yield f"data: {json.dumps(final_payload, default=str)}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get(
    "/api/german/warehouse",
    tags=["German Warehouse"],
    response_model=GermanWarehouseResponse,
    summary="German Warehouse Intelligence",
    description="""
    Query German warehouse data using natural language (English or German).
    
    This endpoint provides AI-powered access to the German military logistics warehouse database,
    including quarterly inventory data and forecasts.
    
    **Available Data:**
    - WAREHOUSE: Master data with article descriptions, numbers, and monthly forecasts
    - WAREHOUSE_Q424, Q125, Q225, Q325: Quarterly warehouse data
    
    **Example Questions:**
    - "Wie viele Artikel haben wir im Lager?" (How many articles do we have in the warehouse?)
    - "Welche Artikel haben die höchste monatliche Prognose?"
    - "Show me articles with monthly forecast over 1000"
    - "What is the total monthly forecast?"
    
    **Example URL:**
    ```
    GET /api/german/warehouse?question=Wie viele Artikel haben wir im Lager?&model=gpt-4o
    ```
    
    **Query Parameters:**
    - `question`: Your question about warehouse data (required)
    - `model`: AI model to use - gpt-4o, gpt-5, or gpt-35-turbo (optional, default: gpt-4o)
    """
)
async def german_warehouse_chat(
    question: str = Query(
        ...,
        description="Question about warehouse data in English or German",
        example="Wie viele Artikel haben wir im Lager?"
    ),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Chat with German warehouse database using natural language.
    """
    try:
        # Call the german_agent final_result function
        result = final_result(
            prompt=question,
            model_name=model
        )
        
        response_data = {
            "question": question,
            "response": result.get("response", "No response generated"),
            "sql_query": result.get("sql_query"),
            "row_count": result.get("row_count"),
            "results": result.get("results", []),
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use JSONResponse to handle datetime serialization
        return JSONResponse(content=json.loads(json.dumps(response_data, default=str)))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying warehouse data: {str(e)}"
        )


# ============================================================================
# Swedish Supplier Data Endpoints
# ============================================================================

@app.get("/api/sweden/suppliers/stream",
    tags=["Swedish Suppliers"])
async def sweden_suppliers_stream(
    query: str = Query(..., description="Natural language query about Swedish supplier data"),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Get streaming response about Swedish supplier and delivery data from HACKATHON_USR.GE_M table.
    
    **Query Examples:**
    - "Which suppliers have the highest combined scores?"
    - "Show me suppliers with problem lots"
    - "What is the average delivery delay by supplier?"
    - "Which suppliers require rework most frequently?"
    - "Show suppliers in Stockholm area"
    """
    try:
        async def event_generator():
            # Get streaming response from Swedish agent
            result = sweden_agent.final_result(prompt=query, model_name=model, stream=True)
            
            # Accumulate the full response
            full_response = ""
            
            # Stream the AI response chunks
            for chunk in result["stream"]:
                chunk_dict = chunk.to_dict()
                choices = chunk_dict.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        full_response += content
                        payload = {
                            "type": "content",
                            "content": content
                        }
                        yield f"data: {json.dumps(payload, default=str)}\n\n"
            
            # Send final metadata with complete response
            final_payload = {
                "type": "metadata",
                "query": query,
                "response": full_response,
                "sql_query": result.get("sql_query"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])[:10],  # Limit results in stream
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_payload, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying Swedish supplier data: {str(e)}"
        )


@app.get("/api/sweden/suppliers",
    tags=["Swedish Suppliers"])
async def sweden_suppliers(
    query: str = Query(..., description="Natural language query about Swedish supplier data"),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Get non-streaming response about Swedish supplier and delivery data from HACKATHON_USR.GE_M table.
    
    **Query Examples:**
    - "Which suppliers have the highest combined scores?"
    - "Show me suppliers with problem lots"
    - "What is the average delivery delay by supplier?"
    - "Which suppliers require rework most frequently?"
    - "Show suppliers in Stockholm area"
    
    **Returns:**
    - query: Your original query
    - response: AI-generated answer to your query
    - sql_query: The SQL query that was executed
    - row_count: Number of rows returned from database
    - results: Sample of database results (limited to 50 rows)
    - model_used: AI model that was used
    - timestamp: When the query was processed
    """
    try:
        # Get non-streaming response from Swedish agent
        result = sweden_agent.final_result(prompt=query, model_name=model, stream=False)
        
        response_data = {
            "query": query,
            "response": result.get("response", "No response generated"),
            "sql_query": result.get("sql_query"),
            "row_count": result.get("row_count", 0),
            "results": result.get("results", []),
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use JSONResponse with JSON serialization to handle datetime objects
        return JSONResponse(content=json.loads(json.dumps(response_data, default=str)))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying Swedish supplier data: {str(e)}"
        )

# ============================================================================
# UK Demand Tracking Endpoints
# ============================================================================

@app.get(
    "/api/uk/demand/stream",
    tags=["UK Demand"],
    summary="UK Demand Tracking Intelligence (Streaming)",
    description="""
    Query UK supply network demand and delivery tracking data using natural language with Server-Sent Events streaming.
    
    This endpoint provides AI-powered access to the UK military supply chain demand database
    with real-time streaming responses.
    
    **Available Data:**
    - SUP_NET_UK.DEMAND: UK supply network demand tracking with delivery checkpoints
    
    **Example Questions:**
    - "What are the current demands by theatre?"
    - "Show me demands with high priority"
    - "Which units have the most pending demands?"
    - "What is the average delivery time by theatre?"
    - "Show demands that are delayed"
    
    **Example URL:**
    ```
    GET /api/uk/demand/stream?question=What are the current demands by theatre?&model=gpt-4o
    ```
    
    **Response Format:**
    Server-Sent Events (SSE) stream with:
    - Text chunks as they're generated
    - Final JSON payload with complete results
    """
)
async def uk_demand_stream(
    question: str = Query(
        ...,
        description="Question about UK demand and delivery tracking",
        example="What are the current demands by theatre?"
    ),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Stream chat responses with UK demand tracking database.
    """
    try:
        async def event_generator():
            # Get streaming response from UK agent
            result = uk_agent.final_result(prompt=question, model_name=model, stream=True)
            
            # Check for errors
            if result.get("error"):
                payload = {
                    "type": "error",
                    "message": result.get("response", "Unknown error")
                }
                yield f"data: {json.dumps(payload, default=str)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Accumulate the full response
            full_response = ""
            
            # Stream the AI response chunks
            for chunk in result["stream"]:
                chunk_dict = chunk.to_dict()
                choices = chunk_dict.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        full_response += content
                        payload = {
                            "type": "content",
                            "content": content
                        }
                        yield f"data: {json.dumps(payload, default=str)}\n\n"
            
            # Send final metadata with complete response
            final_payload = {
                "type": "metadata",
                "question": question,
                "response": full_response,
                "sql_query": result.get("sql_query"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])[:10],  # Limit results in stream
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_payload, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying UK demand data: {str(e)}"
        )


@app.get(
    "/api/uk/demand",
    tags=["UK Demand"],
    response_model=UKDemandResponse,
    summary="UK Demand Tracking Intelligence",
    description="""
    Query UK supply network demand and delivery tracking data using natural language.
    
    This endpoint provides AI-powered access to the UK military supply chain demand database,
    including demand tracking, issue tracking, and delivery checkpoint data.
    
    **Available Data:**
    - SUP_NET_UK.DEMAND: UK supply network demand tracking with delivery checkpoints
    
    **Example Questions:**
    - "What are the current demands by theatre?"
    - "Show me demands with high priority"
    - "Which units have the most pending demands?"
    - "What is the average delivery time by theatre?"
    - "Show demands that are delayed"
    - "Which items are most frequently demanded?"
    
    **Example URL:**
    ```
    GET /api/uk/demand?question=What are the current demands by theatre?&model=gpt-4o
    ```
    
    **Query Parameters:**
    - `question`: Your question about UK demand data (required)
    - `model`: AI model to use - gpt-4o, gpt-5, or gpt-35-turbo (optional, default: gpt-4o)
    """
)
async def uk_demand_chat(
    question: str = Query(
        ...,
        description="Question about UK demand and delivery tracking",
        example="What are the current demands by theatre?"
    ),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Chat with UK demand tracking database using natural language.
    """
    try:
        # Call the uk_agent final_result function
        result = uk_agent.final_result(
            prompt=question,
            model_name=model
        )
        
        response_data = {
            "question": question,
            "response": result.get("response", "No response generated"),
            "sql_query": result.get("sql_query"),
            "row_count": result.get("row_count"),
            "results": result.get("results", []),
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use JSONResponse to handle datetime serialization
        return JSONResponse(content=json.loads(json.dumps(response_data, default=str)))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying UK demand data: {str(e)}"
        )


# ============================================================================
# Finland Garrison Health Endpoints
# ============================================================================

@app.get(
    "/api/finland/health/stream",
    tags=["Finland Garrison Health"],
    summary="Finland Garrison Health Monitoring (Streaming)",
    description="""
    Query Finnish Defense Forces garrison health and epidemic monitoring data using natural language with Server-Sent Events streaming.
    
    This endpoint provides AI-powered access to the Finland health database for epidemic detection,
    risk assessment, and countermeasure recommendations with real-time streaming responses.
    
    **Available Data:**
    - FI_DISEASE_INFO: Disease catalog with epidemic thresholds and prevention measures
    - FI_GARRISON_INFO: Finnish Defense Forces garrison information
    - FI_CONSCRIPT_ORGANIZATION: Conscript assignments and home locations
    - FI_CONSCRIPT_HEALTH: Individual health records and disease tracking
    - FI_THL_DISEASE_CASES: Regional weekly disease statistics from THL
    - FI_MUNICIPALITY_MAPPING: Municipality to wellbeing county mapping
    
    **Example Questions:**
    - "Is there a possibility for an epidemic at Kainuun prikaati?"
    - "What is the current health status of conscripts?"
    - "Show disease trends in garrisons over the last month"
    - "Which diseases are trending in specific regions?"
    - "What countermeasures should be taken for flu outbreak?"
    - "How could a disease outbreak affect training?"
    - "Should conscripts be prevented from weekend visits?"
    - "What is the weekend travel risk based on home regions?"
    
    **Example URL:**
    ```
    GET /api/finland/health/stream?question=Is there an epidemic at Kainuun prikaati?&model=gpt-4o
    ```
    
    **Response Format:**
    Server-Sent Events (SSE) stream with:
    - Text chunks as they're generated
    - Final JSON payload with complete results including SQL query and database results
    """
)
async def finland_health_stream(
    question: str = Query(
        ...,
        description="Question about Finnish garrison health and epidemic monitoring",
        example="Is there a possibility for an epidemic?"
    ),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Streaming endpoint for Finland garrison health queries.
    Returns Server-Sent Events with AI-generated responses.
    """
    try:
        async def event_generator():
            # Get streaming response from Finland agent
            result = agent_finland.final_result(prompt=question, model_name=model, stream=True)
            
            # Accumulate the full response
            full_response = ""
            
            # Stream the AI response chunks
            for chunk in result["stream"]:
                chunk_dict = chunk.to_dict()
                choices = chunk_dict.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        full_response += content
                        payload = {
                            "type": "content",
                            "content": content
                        }
                        yield f"data: {json.dumps(payload, default=str)}\n\n"
            
            # Send final metadata with complete response
            final_payload = {
                "type": "metadata",
                "question": question,
                "response": full_response,
                "sql_query": result.get("sql_query"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])[:10],  # Limit results in stream
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_payload, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying Finland health data: {str(e)}"
        )


@app.get(
    "/api/finland/health",
    tags=["Finland Garrison Health"],
    summary="Finland Garrison Health Monitoring (Non-Streaming)",
    description="""
    Query Finnish Defense Forces garrison health and epidemic monitoring data using natural language.
    
    This endpoint provides AI-powered access to the Finland health database for epidemic detection,
    risk assessment, and countermeasure recommendations.
    
    **Available Data:**
    - FI_DISEASE_INFO: Disease catalog with epidemic thresholds and prevention measures
    - FI_GARRISON_INFO: Finnish Defense Forces garrison information
    - FI_CONSCRIPT_ORGANIZATION: Conscript assignments and home locations (3000 conscripts)
    - FI_CONSCRIPT_HEALTH: Individual health records and disease tracking (3000 records)
    - FI_THL_DISEASE_CASES: Regional weekly disease statistics from THL (2025-2026)
    - FI_MUNICIPALITY_MAPPING: Municipality to wellbeing county mapping
    
    **Example Questions:**
    - "Is there a possibility for an epidemic at Kainuun prikaati?"
    - "What is the current health status of conscripts?"
    - "Show disease trends in garrisons over the last month"
    - "Which diseases are trending in specific regions?"
    - "What countermeasures should be taken for a flu outbreak?"
    - "How could a disease outbreak affect training?"
    - "Should conscripts be prevented from weekend visits?"
    - "What is the weekend travel risk based on home regions?"
    - "Show me units with high infection rates"
    - "What are the symptoms and prevention for Influenssa?"
    
    **Epidemic Thresholds:**
    - Influenssa (Influenza): 30 cases
    - Rinovirus (Rhinovirus): 50 cases
    - Koronavirus (Coronavirus): 20 cases
    - Norovirus: 15 cases
    - Adenovirus: 25 cases
    
    **Returns:**
    - question: Your original question
    - response: AI-generated comprehensive answer with recommendations
    - sql_query: The SQL query that was executed
    - row_count: Number of rows returned from database
    - results: Database results (limited to 50 rows)
    - model_used: AI model that was used
    - timestamp: When the query was processed
    """
)
async def finland_health(
    question: str = Query(
        ...,
        description="Question about Finnish garrison health and epidemic monitoring",
        example="Is there a possibility for an epidemic?"
    ),
    model: str = Query(
        "gpt-4o",
        description="AI model to use (gpt-4o, gpt-5, gpt-35-turbo)",
        example="gpt-4o"
    )
):
    """
    Non-streaming endpoint for Finland garrison health queries.
    Returns complete JSON response with AI-generated answer and database results.
    """
    try:
        # Call the agent_finland final_result function
        result = agent_finland.final_result(
            prompt=question,
            model_name=model,
            stream=False
        )
        
        response_data = {
            "question": question,
            "response": result.get("response", "No response generated"),
            "sql_query": result.get("sql_query"),
            "row_count": result.get("row_count", 0),
            "results": result.get("results", []),
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use JSONResponse to handle datetime serialization
        return JSONResponse(content=json.loads(json.dumps(response_data, default=str)))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying Finland health data: {str(e)}"
        )


# ============================================================================
# PDF Processing Endpoint
# ============================================================================

import shutil
from uk_agent import pdf_reader
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
@app.post("/pdf_reader", tags=["PDF Processing"])
async def process_pdf_endpoint(file: UploadFile = File(...)):
    """
    Accepts a PDF file, saves it temporarily, extracts and cleans the text,
    and returns the text in a JSON response.
    """
    # Create a secure path for the temporary file
    temp_file_path = os.path.join(TEMP_DIR, file.filename)

    try:
        # Save the uploaded file to the temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the PDF using your function
        extracted_text = pdf_reader(temp_file_path)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF. The document might be image-based or empty.")

        # Prepare the successful response
        response_content = {"filename": file.filename, "text": extracted_text}
        return JSONResponse(status_code=200, content=response_content)

    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Close the uploaded file stream
        await file.close()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
