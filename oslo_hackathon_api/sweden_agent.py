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
    logger.info(f"[SWEDEN_AGENT] Received user query: {user_query}")
    
    schema_context = """
        You are an expert SQL developer working with a SAP HANA database containing supplier and delivery data.

        The database contains 1 table with the following schema:

        Table: HACKATHON_USR.GE_M (Supplier and Delivery Performance Data)
        - RECORD_ID: Unique record identifier
        - SUPPLIER_NAME (STRING): Name of the supplier
        - DELIVERY_NUMBER (STRING): Delivery tracking number
        - EXPECTED_DELIVERY_DATE (DATE): Expected date of delivery
        - ACTUAL_DELIVERY_DATE (DATE): Actual date when delivery occurred
        - PROBLEM_LOT (STRING): Indicates if there was a problem with the lot (values: "Y" or "N")
        - PERFECTION_REVIEW (STRING): Indicates if perfection review was done (values: "Y" or "N")
        - ENGINEERING_CLASS_QUALITY (STRING): Single letter quality classification
        - PART_FAMILY_ME (STRING): Part family classification
        - REWORK_NEEDED (STRING): Indicates if rework was required (values: "YES" or "NO")
        - SUPPLIER_ID (DOUBLE): Numeric supplier identifier
        - SUPPLIER_CODE_SOURCING (STRING): Sourcing code for the supplier
        - GSL (DOUBLE): GSL metric value
        - SITE (STRING): Site location code
        - LOCATION (STRING): City/location name
        - GROUP_NAME (STRING): Supplier group name
        - OPEN_PNS_SOURCING (DOUBLE): Open purchase orders in sourcing
        - COMBINED_SCORE (DOUBLE): Combined performance score
        - SUPPLIER_CAPABILITY_COMPLEXITY (STRING): Capability complexity indicator (values: "YES" or "NO")
        - PCT_DELIVERY_SLIPS (DOUBLE): Percentage of delivery slips
        - GME_SOURCING (STRING): GME sourcing classification
        - SOURCING_APPROVAL_REQD (STRING): Whether sourcing approval is required
        - EXIT_DO_NOT_BID (STRING): Exit/do not bid indicator
        - LONGITUDE (DOUBLE): Geographic longitude coordinate
        - LATITUDE (DOUBLE): Geographic latitude coordinate

        CRITICAL SQL RULES:
        1. NO Parameter Placeholders:
           - NEVER use :variable, :supplier_id, or any bind parameters
           - Use literal values directly in WHERE clauses

        2. NO Placeholder Strings:
           - NEVER use '<SupplierName>', '<Please_enter>', or similar placeholders
           - If a value is in the question, use it directly; otherwise query all records

        3. String Comparisons:
           - For PROBLEM_LOT: use "Y" or "N" (case-sensitive strings)
           - For PERFECTION_REVIEW: use "Y" or "N" (case-sensitive strings)
           - For REWORK_NEEDED: use "YES" or "NO" (case-sensitive strings)
           - For SUPPLIER_CAPABILITY_COMPLEXITY: use "YES" or "NO" (case-sensitive strings)

        4. Date Calculations:
           - To calculate delivery delays: DAYS_BETWEEN("EXPECTED_DELIVERY_DATE", "ACTUAL_DELIVERY_DATE")
           - Positive values mean late delivery, negative means early delivery

        5. Pattern for Highest AND Lowest Queries:
           SAP HANA requires separate CTEs - DO NOT use scalar subqueries in SELECT:
           WITH metrics AS (
             SELECT "SUPPLIER_NAME", AVG("COMBINED_SCORE") AS avg_score
             FROM "HACKATHON_USR"."GE_M"
             GROUP BY "SUPPLIER_NAME"
           ),
           highest AS (SELECT * FROM metrics ORDER BY avg_score DESC LIMIT 1),
           lowest AS (SELECT * FROM metrics ORDER BY avg_score ASC LIMIT 1)
           SELECT h."SUPPLIER_NAME" AS best_supplier, h.avg_score AS best_score,
                  l."SUPPLIER_NAME" AS worst_supplier, l.avg_score AS worst_score
           FROM highest h, lowest l

        6. NO Semicolons:
           - NEVER end SQL with semicolon (;) - causes SAP HANA syntax errors
           - End SQL statements without any terminator

        7. General:
           - Always use double quotes for schema/table/column names
           - Table reference: "HACKATHON_USR"."GE_M"
           - Use LIMIT to restrict results
           - Use NULLIF to avoid division by zero
           - Use aggregation functions (AVG, SUM, COUNT, MAX, MIN) for analytics
           - An important rule of thumb - whenever you can, include lon/lat in your answers, as they are critical for geospatial analysis.
        """
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[SWEDEN_AGENT] Attempt {attempt}/{max_retries}")
            
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
            
            logger.info(f"[SWEDEN_AGENT] Generated SQL: {sql_statement}")
            
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
            
            logger.info(f"[SWEDEN_AGENT] Query executed successfully. Rows returned: {len(results)}")
            logger.info(f"[SWEDEN_AGENT] Column names: {column_names}")
            
            cursor.close()
            conn.close()
            
            # Check if results are null or empty
            if results is None or len(results) == 0:
                logger.warning(f"[SWEDEN_AGENT] Attempt {attempt}/{max_retries}: Query returned no results. Retrying...")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"[SWEDEN_AGENT] All {max_retries} attempts returned no results")
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
            
            logger.info(f"[SWEDEN_AGENT] Formatted {len(formatted_results)} results on attempt {attempt}")
            
            return {
                "query": user_query,
                "sql_generated": sql_statement,
                "row_count": len(results),
                "results": formatted_results,
                "note": "Results limited to 50 rows" if len(results) > 50 else "All results shown",
                "attempts": attempt
            }
            
        except Exception as e:
            logger.error(f"[SWEDEN_AGENT] Attempt {attempt}/{max_retries} - Error executing database query: {str(e)}")
            if attempt < max_retries:
                logger.info(f"[SWEDEN_AGENT] Retrying due to error...")
                continue
            else:
                logger.error(f"[SWEDEN_AGENT] All {max_retries} attempts failed")
                return {"error": f"Error executing database query after {max_retries} attempts: {str(e)}"}
    
def final_prompt_supplier(orig_result):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a supply chain and supplier performance analyst. You have previously asked the following question: '{orig_result['query']}'" +
                    f" The results from the database query are as follows: '{orig_result['results']}'. Based on these results, provide a concise and informative answer to the user's original question." +
                    " Focus on supplier performance, delivery metrics, quality indicators, and actionable insights."
                }
            ]
        }
    ]
    return messages

def final_result(prompt, model_name="gpt-4o", stream=False):
    logger.info(f"[SWEDEN_AGENT] Processing final result with model: {model_name}, stream={stream}")
    result = query_database_with_sql_tool(prompt)
    
    if "error" in result:
        logger.error(f"[SWEDEN_AGENT] Database query failed: {result['error']}")
        return {
            "response": f"Database query failed: {result['error']}",
            "sql_query": None,
            "error": True
        }
    
    logger.info(f"[SWEDEN_AGENT] Generating final response from {result.get('row_count', 0)} database results")
    final_messages = final_prompt_supplier(result)
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
        logger.info(f"[SWEDEN_AGENT] Final response generated successfully (length: {len(response_text)} chars)")
        
        try:
            return {
                "response": response_text,
                "sql_query": result.get("sql_generated"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])
            }
        except Exception as e:
            logger.error(f"[SWEDEN_AGENT] Error processing final response: {str(e)}")
            return {
                "response": f"Error processing final response: {str(e)}",
                "sql_query": result.get("sql_generated"),
                "results": result.get("results", []),
                "error": True
            }
