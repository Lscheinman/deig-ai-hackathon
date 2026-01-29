import os
import requests
import json
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional


def get_ai_core_token():
    """
    Get OAuth2 token from AI Core authentication endpoint.
    
    Returns:
        str: Bearer token for AI Core API calls
    """
    auth_url = os.getenv("AICORE_AUTH_URL")
    client_id = os.getenv("AICORE_CLIENT_ID")
    client_secret = os.getenv("AICORE_CLIENT_SECRET")
    
    if not all([auth_url, client_id, client_secret]):
        raise ValueError("AI Core credentials not found in environment variables")
    
    # Request token
    token_response = requests.post(
        f"{auth_url}/oauth/token",
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"}
    )
    token_response.raise_for_status()
    
    return token_response.json()["access_token"]


def predict_with_rpt1(
    rows: List[Dict],
    index_column: Optional[str] = None,
    target_columns: Optional[List[Dict]] = None,
    data_schema: Optional[Dict] = None,
    model_size: str = "small",
    deployment_url: Optional[str] = None
) -> Optional[Dict]:
    """
    Make predictions using SAP-RPT-1 Model deployed on AI Core.
    
    Args:
        rows (list): List of dictionaries representing data rows
                    Context rows should have complete data
                    Query rows should have '[PREDICT]' for values to predict
        index_column (str, optional): Column name to use as row identifier
        target_columns (list, optional): List of dicts with target column configuration
                    Example: [{"name": "COSTCENTER", "prediction_placeholder": "[PREDICT]", "task_type": "classification"}]
                    If not provided, will auto-detect from rows
        data_schema (dict, optional): Schema defining column data types
                    Example: {"PRODUCT": {"dtype": "string"}, "PRICE": {"dtype": "numeric"}}
        model_size (str, optional): Model size to use - "small" or "large". Default is "small"
        deployment_url (str, optional): AI Core deployment URL. If not provided, uses RPT_SMALL_URL or RPT_LARGE_URL based on model_size
    
    Returns:
        dict: Prediction results from the API
    
    Example:
        rows = [
            {"ID": "35", "PRODUCT": "Couch", "PRICE": 999.99, "COSTCENTER": "[PREDICT]"},
            {"ID": "44", "PRODUCT": "Office Chair", "PRICE": 150.8, "COSTCENTER": "Office Furniture"}
        ]
        target_columns = [{"name": "COSTCENTER", "prediction_placeholder": "[PREDICT]", "task_type": "classification"}]
        result = predict_with_rpt1(rows, index_column="ID", target_columns=target_columns, model_size="large")
    """
    
    # Get deployment URL based on model size
    if not deployment_url:
        if model_size.lower() == "small":
            deployment_url = os.getenv("RPT_SMALL_URL")
        elif model_size.lower() == "large":
            deployment_url = os.getenv("RPT_LARGE_URL")
        else:
            raise ValueError(f"Invalid model_size '{model_size}'. Must be 'small' or 'large'")
    
    if not deployment_url:
        raise ValueError(f"Deployment URL not found. Please set RPT_{model_size.upper()}_URL in environment variables")
    
    # Ensure URL ends with /predict
    if not deployment_url.endswith('/predict'):
        deployment_url = deployment_url.rstrip('/') + '/predict'
    
    # Get AI Core token
    try:
        auth_token = get_ai_core_token()
    except Exception as e:
        print(f"Failed to get AI Core token: {e}")
        return None
    
    # Auto-detect target columns if not provided
    if not target_columns:
        target_columns = []
        if rows:
            for key, value in rows[0].items():
                if value == "[PREDICT]":
                    target_columns.append({
                        "name": key,
                        "prediction_placeholder": "[PREDICT]",
                        "task_type": "classification"  # default, could be inferred better
                    })
    
    # Prepare request payload
    payload = {
        "rows": rows,
        "prediction_config": {
            "target_columns": target_columns
        }
    }
    
    if index_column:
        payload["index_column"] = index_column
    
    if data_schema:
        payload["data_schema"] = data_schema
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "AI-Resource-Group": os.getenv("AICORE_RESOURCE_GROUP", "default"),
        "Content-Type": "application/json"
    }
    
    # Debug logging (set DEBUG_RPT=false to disable)
    debug_mode = "false"
    
    try:
        # Validate data
        if not rows:
            print("Error: No rows provided")
            return None
        
        if not target_columns or len(target_columns) == 0:
            print("Error: No target columns detected. Ensure some values are '[PREDICT]'")
            return None
        
        # Make API request
        print(f"Using RPT-1 {model_size.upper()} model...")
        print(f"Deployment URL: {deployment_url}")
        print(f"Payload preview: {len(rows)} rows, targets: {[tc.get('name') for tc in target_columns]}")
        
        if debug_mode:
            print(f"Full payload being sent:")
            print(json.dumps(payload, indent=2))
            print(f"Headers (auth masked):")
            print(json.dumps({k: v if k != 'Authorization' else 'Bearer ***' for k, v in headers.items()}, indent=2))
        
        response = requests.post(deployment_url, headers=headers, json=payload, timeout=60)
        
        # Check for rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 'unknown')
            print(f"Rate limit exceeded. Retry after {retry_after} seconds.")
            return None
        
        # Check for service unavailability
        if response.status_code == 503:
            retry_after = response.headers.get('Retry-After', 'unknown')
            print(f"Service unavailable. Retry after {retry_after} seconds.")
            return None
        
        # Check response status and provide detailed error info
        if response.status_code != 200:
            error_detail = {
                "status_code": response.status_code,
                "url": deployment_url,
                "response_text": response.text,
                "headers": dict(response.headers)
            }
            print(f"API request failed with status {response.status_code}")
            print(f"Error details: {json.dumps(error_detail, indent=2)}")
            
            # Try to parse error response
            try:
                error_json = response.json()
                print(f"Error response: {json.dumps(error_json, indent=2)}")
            except:
                pass
            
            return None
        
        # Return parsed response
        result = response.json()
        print(f"Prediction successful, received response")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"API request exception: {type(e).__name__}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
            try:
                print(f"Response JSON: {json.dumps(e.response.json(), indent=2)}")
            except:
                pass
        return None
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_from_file(
    file_path: Union[str, Path],
    index_column: Optional[str] = None,
    target_columns: Optional[List[Dict]] = None,
    data_schema: Optional[Dict] = None,
    model_size: str = "small",
    deployment_url: Optional[str] = None
) -> Optional[Dict]:
    """
    Make predictions using SAP-RPT-1 Model from a CSV or JSON file.
    
    Args:
        file_path (str or Path): Path to the CSV or JSON file containing the data
        index_column (str, optional): Column name to use as row identifier
        target_columns (list, optional): List of dicts with target column configuration
                    Example: [{"name": "COSTCENTER", "prediction_placeholder": "[PREDICT]", "task_type": "classification"}]
                    If not provided, will auto-detect from data
        data_schema (dict, optional): Schema defining column data types
                    Example: {"PRODUCT": {"dtype": "string"}, "PRICE": {"dtype": "numeric"}}
        model_size (str, optional): Model size to use - "small" or "large". Default is "small"
        deployment_url (str, optional): AI Core deployment URL. If not provided, uses RPT_SMALL_URL or RPT_LARGE_URL based on model_size
    
    Returns:
        dict: Prediction results from the API
    
    Example:
        # For CSV file
        result = predict_from_file("data.csv", index_column="ID", model_size="large")
        
        # For JSON file with custom target columns
        target_cols = [{"name": "PRICE", "prediction_placeholder": "[PREDICT]", "task_type": "regression"}]
        result = predict_from_file("data.json", index_column="ID", target_columns=target_cols)
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        # Read CSV file
        df = pd.read_csv(file_path)
        # Convert DataFrame to list of dictionaries
        rows = df.to_dict('records')
        
    elif file_path.suffix.lower() == '.json':
        # Read JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            # Check if it has a 'rows' or 'data' key
            if 'rows' in data:
                rows = data['rows']
            elif 'data' in data:
                rows = data['data']
            else:
                # Convert dict to DataFrame and then to rows
                df = pd.DataFrame(data)
                rows = df.to_dict('records')
        else:
            raise ValueError("JSON file must contain a list of objects or a dictionary")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .json")
    
    if not rows:
        raise ValueError("No data found in file")
    
    print(f"Loaded {len(rows)} rows from {file_path}")
    
    # Make prediction using the existing function
    return predict_with_rpt1(
        rows=rows,
        index_column=index_column,
        target_columns=target_columns,
        data_schema=data_schema,
        model_size=model_size,
        deployment_url=deployment_url
    )
