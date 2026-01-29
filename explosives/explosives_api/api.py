"""
Explosives Intelligence API
===========================
FastAPI backend for the Explosives Intelligence Agent.
Provides REST endpoints for querying explosives data and AI-powered analysis.

Run with: uvicorn api:app --reload
Access Swagger UI at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import json
import asyncio

# Import the agent module
from agent import (
    run_explosives_agent,
    get_compatibility_info_tool,
    get_material_inventory_tool,
    query_database_with_sql_tool,
    list_all_materials_tool,
    get_storage_summary_tool,
    TOOLS,
    AVAILABLE_FUNCTIONS
)

# Import for streaming agent
from gen_ai_hub.proxy.native.openai import chat

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Explosives Intelligence API",
    description="""
    **Explosives Safety & Inventory Management System**
    
    This API provides intelligent access to explosives data including:
    - Material compatibility analysis
    - Inventory tracking across storage locations
    - Natural language database queries
    - Safety compliance checking
    - Storage optimization insights
    
    The system uses AI to interpret natural language questions and provide
    expert-level guidance on explosives safety and management.
    """,
    version="1.0.0",
    contact={
        "name": "Explosives Intelligence Team",
        "email": "explosives@example.com",
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

class AgentQueryRequest(BaseModel):
    """Request model for AI agent queries."""
    question: str = Field(
        ...,
        description="Natural language question about explosives",
        example="Can I store materials from compatibility group D and F together?"
    )
    model: Optional[str] = Field(
        default="gpt-5",
        description="AI model to use (gpt-5, gpt-4o, gpt-35-turbo)",
        example="gpt-5"
    )
    max_iterations: Optional[int] = Field(
        default=10,
        description="Maximum number of tool calls the agent can make",
        ge=1,
        le=20
    )


class AgentQueryResponse(BaseModel):
    """Response model for AI agent queries."""
    model_config = ConfigDict(protected_namespaces=())
    
    question: str = Field(..., description="The original question asked")
    answer: str = Field(..., description="The AI agent's answer")
    timestamp: str = Field(..., description="Timestamp of the response")
    model_used: str = Field(..., description="AI model that was used")


class MaterialCompatibilityResponse(BaseModel):
    """Response model for material compatibility queries."""
    material_number: int
    material_description: str
    compatibility_group: str
    compatibility_group_description: str
    hazard_division: str
    compatibility_with_other_groups: Dict[str, str]
    compatibility_legend: Dict[str, str]


class InventoryResponse(BaseModel):
    """Response model for inventory queries."""
    material_number: int
    inventory_locations: List[Dict[str, Any]]
    total_quantity_all_locations: int
    total_new_all_locations_kg: float
    number_of_storage_locations: int


class SQLQueryRequest(BaseModel):
    """Request model for natural language SQL queries."""
    query: str = Field(
        ...,
        description="Natural language description of the data you want to retrieve",
        example="Show all materials in compatibility group D with NEW values"
    )


class SQLQueryResponse(BaseModel):
    """Response model for SQL queries."""
    query: str = Field(..., description="Original natural language query")
    sql_generated: str = Field(..., description="Generated SQL statement")
    row_count: int = Field(..., description="Number of rows returned")
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    note: str = Field(..., description="Additional notes about the results")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str


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
        "message": "Explosives Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "agent": "/api/agent/query",
            "compatibility": "/api/materials/{material_number}/compatibility",
            "inventory": "/api/materials/{material_number}/inventory",
            "sql_query": "/api/database/query",
            "materials": "/api/materials",
            "storage": "/api/storage/summary"
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
    "/api/agent/query",
    tags=["AI Agent"],
    response_model=AgentQueryResponse,
    summary="Ask the AI Agent",
    description="""
    Ask the AI agent any question about explosives safety, compatibility, 
    inventory, or storage. The agent will automatically select the appropriate 
    tools to answer your question.
    
    **Example questions:**
    - "What is the compatibility information for material 885600034?"
    - "Can I store materials from compatibility group D and F together?"
    - "Show me all materials with more than 100 units in inventory"
    - "Which storage location has the most explosives?"
    """
)
async def query_agent(request: AgentQueryRequest):
    """
    Query the AI agent with a natural language question.
    
    The agent will analyze your question and use the appropriate tools
    to provide an accurate, safety-focused answer.
    """
    try:
        answer = run_explosives_agent(
            user_question=request.question,
            max_iterations=request.max_iterations,
            model=request.model,
            verbose=False  # Disable verbose output for API
        )
        
        return {
            "question": request.question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "model_used": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get(
    "/api/materials/{material_number}/compatibility",
    tags=["Materials"],
    response_model=MaterialCompatibilityResponse,
    summary="Get Material Compatibility",
    description="""
    Get detailed compatibility information for a specific material including:
    - Compatibility group and description
    - Hazard division
    - Compatibility matrix with all other groups
    - Legend explaining compatibility codes
    """
)
async def get_material_compatibility(
    material_number: int = Path(..., description="Material number to look up", example=885600034)
):
    """Get compatibility information for a specific material."""
    try:
        result = get_compatibility_info_tool(material_number)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get(
    "/api/materials/{material_number}/inventory",
    tags=["Materials"],
    response_model=InventoryResponse,
    summary="Get Material Inventory",
    description="""
    Get detailed inventory information for a specific material across all 
    storage locations including:
    - Quantities at each location
    - Storage location descriptions
    - Total NEW (Net Explosive Weight)
    - Hazard division and compatibility group
    """
)
async def get_material_inventory(
    material_number: int = Path(..., description="Material number to check", example=885600034)
):
    """Get inventory information for a specific material."""
    try:
        result = get_material_inventory_tool(material_number)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post(
    "/api/database/query",
    tags=["Database"],
    response_model=SQLQueryResponse,
    summary="Natural Language Database Query",
    description="""
    Execute a natural language query against the explosives database.
    The system will automatically generate and execute the appropriate SQL.
    
    **Example queries:**
    - "Show all materials in compatibility group D"
    - "Find materials with more than 100kg NEW"
    - "List all storage locations with their total NEW"
    - "What materials are stored in location IGS-01?"
    """
)
async def query_database(request: SQLQueryRequest):
    """
    Execute a natural language query against the database.
    
    The system uses AI to generate optimized SQL queries from your 
    natural language description.
    """
    try:
        result = query_database_with_sql_tool(request.query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get(
    "/api/materials",
    tags=["Materials"],
    summary="List All Materials",
    description="""
    Get a list of all explosive materials in the database with their 
    basic properties including:
    - Material number and description
    - Hazard division
    - Compatibility group
    - Net Explosive Weight (NEW)
    
    Note: Results are limited to 100 materials for performance.
    """
)
async def list_materials():
    """List all explosive materials in the database."""
    try:
        result = list_all_materials_tool()
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get(
    "/api/storage/summary",
    tags=["Storage"],
    summary="Get Storage Summary",
    description="""
    Get a summary of all storage locations including:
    - Number of unique materials stored
    - Total number of items
    - Total NET Explosive Weight (NEW)
    - Storage location descriptions
    
    Results are ordered by total NEW in descending order.
    """
)
async def get_storage_summary():
    """Get a summary of all storage locations."""
    try:
        result = get_storage_summary_tool()
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ============================================================================
# SSE Streaming Endpoint
# ============================================================================

async def agent_stream_generator(user_question: str, model: str = "gpt-5"):
    """
    Generator function that yields Server-Sent Events for agent execution.
    Streams the agent's reasoning process in real-time.
    """
    system_message = """You are an expert explosives safety and inventory management assistant. 
You have access to a comprehensive database of explosive materials, their compatibility groups, 
hazard divisions, and LIVE inventory data from S/4 HANA.

Your role is to:
- Answer questions about explosive materials and their properties
- Provide compatibility information and storage guidance
- Query LIVE inventory levels and storage locations from S/4 HANA
- Calculate IATG-aggregated NEW values for Quantity Distance (QTD) calculations
- Explain safety considerations and compatibility rules
- Generate insights from the explosives database
- Perform 'what-if' analysis for material combinations

IMPORTANT: Inventory data (quantities, storage locations, batches) is fetched LIVE from S/4 HANA,
not from HANA tables. Always use get_material_inventory or get_storage_summary for current stock information.

IATG AGGREGATION: Use calculate_storage_location_qtd for QTD calculations for existing storage locations,
and calculate_materials_aggregation for 'what-if' scenarios when planning to combine specific materials.

Always provide clear, accurate, and safety-focused information. When discussing compatibility,
reference the specific compatibility codes (0, X, 1-7) and explain their meanings."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_question}
    ]
    
    # Send initial status
    yield f"data: {json.dumps({'type': 'status', 'message': 'Starting explosives intelligence agent...', 'timestamp': datetime.now().isoformat()})}\n\n"
    
    await asyncio.sleep(0.3)  # Small delay for UX
    
    for iteration in range(10):
        # Send iteration status
        yield f"data: {json.dumps({'type': 'iteration', 'iteration': iteration + 1, 'message': 'Analyzing your question...', 'timestamp': datetime.now().isoformat()})}\n\n"
        
        # Call the model
        try:
            response = chat.completions.create(
                model_name=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            
            response_dict = response.to_dict()
            response_message = response_dict["choices"][0]["message"]
            tool_calls = response_message.get("tool_calls")
            
            if not tool_calls:
                # No more function calls, send final answer
                final_answer = response_message.get("content")
                
                yield f"data: {json.dumps({'type': 'final_answer', 'content': final_answer, 'timestamp': datetime.now().isoformat()})}\n\n"
                return
            
            messages.append(response_message)
            
            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                # Send tool call start event
                yield f"data: {json.dumps({'type': 'tool_call_start', 'tool_name': function_name, 'parameters': function_args, 'timestamp': datetime.now().isoformat()})}\n\n"
                
                await asyncio.sleep(0.5)  # Delay for UX
                
                # Execute function
                try:
                    function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
                    
                    # Send tool call result
                    result_preview = str(function_response)[:300] + "..." if len(str(function_response)) > 300 else str(function_response)
                    yield f"data: {json.dumps({'type': 'tool_call_result', 'tool_name': function_name, 'success': True, 'result_preview': result_preview, 'result_size': len(json.dumps(function_response)), 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                except Exception as e:
                    function_response = {"error": str(e)}
                    
                    # Send error event
                    yield f"data: {json.dumps({'type': 'tool_call_error', 'tool_name': function_name, 'error': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"
                
                # Add response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
                
                await asyncio.sleep(0.3)
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Agent error: {str(e)}', 'timestamp': datetime.now().isoformat()})}\n\n"
            return
    
    # Max iterations reached
    yield f"data: {json.dumps({'type': 'error', 'message': 'Maximum iterations reached. Could not complete the query.', 'timestamp': datetime.now().isoformat()})}\n\n"


@app.post("/agent/stream")
async def stream_agent_endpoint(
    question: str = Query(..., description="Natural language question about explosives"),
    model: str = Query("gpt-5", description="AI model to use (gpt-5, gpt-4o, gpt-35-turbo)")
):
    """
    Server-Sent Events (SSE) endpoint for streaming agent responses.
    
    Streams real-time events during agent execution:
    - Status updates
    - Tool calls with parameters
    - Tool results
    - Final answer
    
    **Event Types:**
    - `status`: Initial connection and progress updates
    - `iteration`: Each reasoning iteration
    - `tool_call_start`: When a tool is about to be called
    - `tool_call_result`: When a tool returns successfully
    - `tool_call_error`: When a tool encounters an error
    - `final_answer`: The agent's final response
    - `error`: Any unexpected errors
    
    **Client Usage Example (Python):**
    ```python
    import requests
    
    response = requests.post(
        'http://localhost:8000/agent/stream',
        params={'question': 'What is material 885600034?', 'model': 'gpt-5'},
        stream=True
    )
    
    for line in response.iter_lines():
        if line and line.startswith(b'data: '):
            data = json.loads(line[6:])
            print(data['type'], data)
    ```
    
    **Client Usage Example (JavaScript):**
    ```javascript
    const eventSource = new EventSource('/agent/stream?question=What+is+material+885600034&model=gpt-5');
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.type, data);
    };
    ```
    """
    return StreamingResponse(
        agent_stream_generator(question, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx
        }
    )


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
