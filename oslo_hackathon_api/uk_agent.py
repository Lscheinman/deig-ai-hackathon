from pypdf import PdfReader
import os
import logging
from dotenv import load_dotenv
from hdbcli import dbapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

HANA_HOST = os.getenv("HANA_HOST")
HANA_PORT = os.getenv("HANA_PORT")
HANA_USER = os.getenv("HANA_USER")
HANA_PASSWORD = os.getenv("HANA_PASSWORD")

os.environ['AICORE_CLIENT_ID'] = os.getenv("AICORE_CLIENT_ID")
os.environ['AICORE_CLIENT_SECRET'] = os.getenv("AICORE_CLIENT_SECRET")
os.environ['AICORE_AUTH_URL'] = os.getenv("AICORE_AUTH_URL")
os.environ['AICORE_BASE_URL'] = os.getenv("AICORE_BASE_URL")
os.environ['AICORE_RESOURCE_GROUP'] = os.getenv("AICORE_RESOURCE_GROUP")

from gen_ai_hub.proxy.native.openai import chat

def clean_text(text: str) -> str:
    """Replaces common non-ASCII characters with their ASCII equivalents."""
    # Replace Narrow No-Break Space with a regular space
    text = text.replace('\u202f', ' ')
    # Replace various dashes and hyphens with a standard hyphen
    text = text.replace('–', '-').replace('—', '-')
    # Replace "smart" quotes with standard quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    # Replace bullets with asterisks or hyphens
    text = text.replace('•', '*')
    # You can add more replacements here as you find other characters
    return text

def pdf_reader(doc_path: str) -> str:
    """
    Reads a PDF, extracts text from all pages, cleans it, and returns a single string.
    """
    try:
        reader = PdfReader(doc_path)
        full_text = []
        
        for page in reader.pages:
            # Extract text from the page
            page_text = page.extract_text()
            if page_text:
                # Clean the extracted text before appending
                cleaned_page_text = clean_text(page_text)
                full_text.append(cleaned_page_text)
        
        # Join all cleaned page texts into a single string
        return "\n\n".join(full_text)
        
    except Exception as e:
        print(f"Error reading PDF {doc_path}: {e}")
        return "" # Return empty string on failure


# ============================================================================
# UK Demand Database Query Functions
# ============================================================================

def query_database_with_sql_tool(user_query: str, max_retries: int = 5):
    """
    Generate SQL from natural language query and execute it against the HANA database.
    Retries up to max_retries times if the result is null/empty.
    
    Args:
        user_query: Natural language description of the query
        max_retries: Maximum number of retry attempts (default: 5)
        
    Returns:
        Dictionary containing SQL, results, and metadata
    """
    logger.info(f"[UK_AGENT] Received user query: {user_query}")
    
    schema_context = """
        You are an expert SQL developer working with a SAP HANA database containing UK military supply chain demand fulfillment and delivery tracking data.

        The database contains 1 table with the following schema:

        Table: SUP_NET_UK.DEMAND (Analytical Record - Demand Fulfillment Tracking)
        
        TABLE DESCRIPTION:
        The Analytical Record table contains a list of demands and the information held about their fulfillment. 
        A demand represents a request from a military unit for material needed to fulfill its mission. Materials 
        might be consumable materials like food, fuel or ammunition, equipment to be used by the unit, or spare 
        parts needed to maintain equipment in an operational condition. The table contains information about the 
        material demanded, the unit that demanded it, and the dates at which the consignment created to fulfill 
        the demand passed through checkpoints on its journey from the supplying location to the demanding unit.

        COLUMNS:
        - Demand_ID (DOUBLE): A unique identifier of a Demand record (integer)
        - Demand_Unit (NVARCHAR(5000)): The name of the unit placing the demand
        - Demand_Theatre (NVARCHAR(5000)): The theatre in which the unit placing the demand is operating
        - Demand_Priority (NVARCHAR(5000)): Priority of the demand based on material importance to operations and current inventory levels
        - Demand_Date (NVARCHAR(5000)): Date and time when the demand was created by the unit (excel datetime value)
        - Demand_NSN (DOUBLE): NATO Stock Number (NSN) identifying the material demanded by the unit
        - Demand_Item_Name (NVARCHAR(5000)): Textual description of the demanded item from the material master record
        - Demand_Qty (DOUBLE): Quantity of material demanded
        - Req_Delivery_Date (NVARCHAR(5000)): Date and time by which the unit requires material delivery (excel datetime value)
        - Issue_ID (NVARCHAR(5000)): Unique identifier for the warehouse order to pick and pack the demanded material
        - Issue_Date (NVARCHAR(5000)): Date and time when the warehouse order was created
        - Issue_NSN (DOUBLE): NATO Stock Number of actual material issued from warehouse (may differ from Demand_NSN if valid alternative)
        - Issue_Item_Name (NVARCHAR(5000)): Textual description of the issued item from the material master record
        - Issue_Qty (DOUBLE): Quantity issued (may be less than Demand_Qty if partial fulfillment)
        - Package_ID (NVARCHAR(5000)): Unique identifier for the package containing the demanded material for transport
        - Package_Created_Date (NVARCHAR(5000)): Date and time package was created and ready for collection
        - Departed_Depot_Date (NVARCHAR(5000)): Date and time package was collected from the supply warehouse
        - Arrived_POE_Date (NVARCHAR(5000)): Date and time package arrived at Port of Embarkation (POE) ready for loading
        - Departed_POE_Date (NVARCHAR(5000)): Date and time package departed from Port of Embarkation (POE)
        - Arrived_POD_Date (NVARCHAR(5000)): Date and time package arrived at Port of Disembarkation (POD) in the demanding unit's theatre
        - Receipted_Date (NVARCHAR(5000)): Date and time package was receipted by the unit that placed the demand
        - Last_Known_Checkpoint (NVARCHAR(5000)): Name of checkpoint that last reported package location
        - Last_Known_Checkpoint_Date (NVARCHAR(5000)): Date and time of last checkpoint report between warehouse and demanding unit
        - Current_Date (NVARCHAR(5000)): Current date reference
        - Late_Status(NVARCHAR(5000)): The status of whether the demand is predicted to be late, based on the output of the machine learning model. Can take values of Predicted Late and Predicted On Time
        - Late_Percent (DOUBLE): The percentage likelihood of whether the demand will be late, based on the output of the machine learning model. A percentage closer to 1 indicates a higher likelihood the demand will be late. A percentage of 1 indicates the demand is already late.

        SUPPLY CHAIN FLOW:
        Demand Created → Warehouse Order (Issue) → Package Created → Departed Depot → 
        Arrived POE → Departed POE → Arrived POD → Receipted by Unit
        
        KEY CONCEPTS:
        - Demand vs Issue: Issue_NSN may differ from Demand_NSN if a valid alternative material is provided
        - Partial Fulfillment: Issue_Qty may be less than Demand_Qty (best offer from warehouse)
        - Priority Levels: Based on operational importance and current unit inventory levels
        - Checkpoints: Track package movement from supply warehouse to demanding unit

        CRITICAL SQL RULES:
        1. NO Parameter Placeholders:
           - NEVER use :variable, :demand_id, or any bind parameters
           - Use literal values directly in WHERE clauses

        2. NO Placeholder Strings:
           - NEVER use '<DemandID>', '<Please_enter>', or similar placeholders
           - If a value is in the question, use it directly; otherwise query all records

        3. String Comparisons:
           - Date fields are stored as NVARCHAR, use string comparisons or TO_DATE() for date operations
           - For partial matches on text fields, use LIKE with wildcards

        4. Date Calculations:
           - Date fields are NVARCHAR, convert using TO_DATE() before date arithmetic
           - To calculate delivery delays: DAYS_BETWEEN(TO_DATE("Req_Delivery_Date", 'YYYY-MM-DD'), TO_DATE("Receipted_Date", 'YYYY-MM-DD'))
           - Positive values mean late delivery, negative means early delivery

        5. Tracking Supply Chain Flow:
           - Supply chain flow: Demand Created → Issue (Warehouse Order) → Package Created → Departed Depot → Arrived POE → Departed POE → Arrived POD → Receipted by Unit
           - Use checkpoint dates to track package progress through supply chain
           - Last_Known_Checkpoint shows the most recent checkpoint that reported package location
           - Compare Demand_NSN with Issue_NSN to identify alternative materials
           - Compare Demand_Qty with Issue_Qty to identify partial fulfillments

        6. Pattern for Highest AND Lowest Queries:
           SAP HANA requires separate CTEs - DO NOT use scalar subqueries in SELECT:
           WITH metrics AS (
             SELECT "Demand_Unit", AVG("Demand_Qty") AS avg_qty
             FROM "SUP_NET_UK"."DEMAND"
             GROUP BY "Demand_Unit"
           ),
           highest AS (SELECT * FROM metrics ORDER BY avg_qty DESC LIMIT 1),
           lowest AS (SELECT * FROM metrics ORDER BY avg_qty ASC LIMIT 1)
           SELECT h."Demand_Unit" AS highest_unit, h.avg_qty AS highest_qty,
                  l."Demand_Unit" AS lowest_unit, l.avg_qty AS lowest_qty
           FROM highest h, lowest l

        7. NO Semicolons:
           - NEVER end SQL with semicolon (;) - causes SAP HANA syntax errors
           - End SQL statements without any terminator

        8. General:
           - Always use double quotes for schema/table/column names
           - Table reference: "SUP_NET_UK"."DEMAND"
           - Use LIMIT to restrict results
           - Use NULLIF to avoid division by zero
           - Use aggregation functions (AVG, SUM, COUNT, MAX, MIN) for analytics
           - Track demand fulfillment rate by comparing Demand_Qty with Issue_Qty
           - Identify alternative materials by comparing Demand_NSN with Issue_NSN
           - Monitor delivery performance by analyzing checkpoint date progression
           - Calculate transit times between checkpoints for performance analysis
           - Filter by Demand_Theatre to analyze theatre-specific operations
           - Use Demand_Priority to identify critical/urgent demands
        """
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[UK_AGENT] Attempt {attempt}/{max_retries}")
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": schema_context}]},
                {"role": "user", "content": [{"type": "text", "text": f"Generate SQL for: {user_query}\nReturn ONLY the SQL statement."}]}
            ]
            
            response = chat.completions.create(model_name="gpt-35-turbo", messages=messages)
            sql_statement = response.to_dict()["choices"][0]["message"]["content"].strip()
            
            # Clean SQL
            if sql_statement.startswith("```sql"):
                sql_statement = sql_statement[6:]
            if sql_statement.startswith("```"):
                sql_statement = sql_statement[3:]
            if sql_statement.endswith("```"):
                sql_statement = sql_statement[:-3]
            sql_statement = sql_statement.strip()
            
            # Remove trailing semicolon (SAP HANA doesn't like it in programmatic execution)
            if sql_statement.endswith(";"):
                sql_statement = sql_statement[:-1].strip()
            
            logger.info(f"[UK_AGENT] Generated SQL: {sql_statement}")
            
            conn = dbapi.connect(
                address=HANA_HOST,
                port=HANA_PORT,
                user=HANA_USER,
                password=HANA_PASSWORD
            )
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            logger.info(f"[UK_AGENT] Query executed successfully. Rows returned: {len(results)}")
            logger.info(f"[UK_AGENT] Column names: {column_names}")
            
            cursor.close()
            conn.close()
            
            # Check if results are null or empty
            if results is None or len(results) == 0:
                logger.warning(f"[UK_AGENT] Attempt {attempt}/{max_retries}: Query returned no results. Retrying...")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"[UK_AGENT] All {max_retries} attempts returned no results")
                    return {
                        "query": user_query,
                        "sql_generated": sql_statement,
                        "row_count": 0,
                        "results": [],
                        "note": f"No results found after {max_retries} attempts"
                    }
            
            # Format results as list of dictionaries
            formatted_results = []
            for row in results[:50]:
                formatted_results.append(dict(zip(column_names, row)))
            
            logger.info(f"[UK_AGENT] Formatted {len(formatted_results)} results on attempt {attempt}")
            
            return {
                "query": user_query,
                "sql_generated": sql_statement,
                "row_count": len(results),
                "results": formatted_results,
                "note": "Results limited to 50 rows" if len(results) > 50 else "All results shown",
                "attempts": attempt
            }
            
        except Exception as e:
            logger.error(f"[UK_AGENT] Attempt {attempt}/{max_retries} - Error executing database query: {str(e)}")
            if attempt < max_retries:
                logger.info(f"[UK_AGENT] Retrying due to error...")
                continue
            else:
                logger.error(f"[UK_AGENT] All {max_retries} attempts failed")
                return {"error": f"Error executing database query after {max_retries} attempts: {str(e)}"}
    
def final_prompt_demand(orig_result):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a UK military supply chain and logistics specialist focusing on demand fulfillment and delivery performance analysis. " +
                    f"You have previously asked the following question: '{orig_result['query']}'" +
                    f" The results from the database query are as follows: '{orig_result['results']}'. " +
                    "Based on these results, provide a concise and informative answer to the user's original question. " +
                    "Focus on demand fulfillment rates (comparing demanded vs issued quantities), delivery performance (checkpoint progression and timing), " +
                    "material substitutions (when Issue_NSN differs from Demand_NSN), partial fulfillments, supply chain bottlenecks, " +
                    "theatre-specific performance, priority-based analysis, and actionable insights for improving military logistics operations."
                }
            ]
        }
    ]
    return messages

def final_result(prompt, model_name="gpt-4o", stream=False):
    logger.info(f"[UK_AGENT] Processing final result with model: {model_name}, stream={stream}")
    result = query_database_with_sql_tool(prompt)
    
    if "error" in result:
        logger.error(f"[UK_AGENT] Database query failed: {result['error']}")
        return {
            "response": f"Database query failed: {result['error']}",
            "sql_query": None,
            "error": True
        }
    
    logger.info(f"[UK_AGENT] Generating final response from {result.get('row_count', 0)} database results")
    final_messages = final_prompt_demand(result)
    response = chat.completions.create(model_name=model_name, messages=final_messages, stream=stream)
    
    if stream:
        # Return generator and metadata for streaming
        return {
            "stream": response,
            "sql_query": result.get("sql_generated"),
            "row_count": result.get("row_count", 0),
            "results": result.get("results", [])
        }
    else:
        response_text = response.to_dict()["choices"][0]["message"]["content"].strip()
        logger.info(f"[UK_AGENT] Final response generated successfully (length: {len(response_text)} chars)")
        
        try:
            return {
                "response": response_text,
                "sql_query": result.get("sql_generated"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])
            }
        except Exception as e:
            logger.error(f"[UK_AGENT] Error processing final response: {str(e)}")
            return {
                "response": f"Error processing final response: {str(e)}",
                "sql_query": result.get("sql_generated"),
                "results": result.get("results", []),
                "error": True
            }