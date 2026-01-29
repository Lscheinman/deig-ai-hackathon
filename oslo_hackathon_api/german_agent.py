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
    logger.info(f"[GERMAN_AGENT] Received user query: {user_query}")
    
    schema_context = """
        You are an expert SQL developer working with a SAP HANA database schema named AI_HACKATHON.

        The database contains 5 tables with the following schema:

        1. Table: AI_HACKATHON.WAREHOUSE (Master Data)
        - Artikelbezeichnung (NVARCHAR(500)): Material description
        - Artikelnummer (NVARCHAR(50), PRIMARY KEY): Material number
        - Einheit (NVARCHAR(50)): Unit of measure
        - Bündnis_Identnumer (NVARCHAR(50)): NATO Stock Number
        - Minimaler_Sicherheitsbestand (DOUBLE): Minimum safety stock
        - Maximaler_Sicherheitsbestand (DOUBLE): Maximum safety stock
        - Frei_verwendbar (DOUBLE): Available stock
        - Monatliche_Prognose_APO_ (DOUBLE): Monthly forecast from APO

        2. Four Quarterly Tables: AI_HACKATHON.WAREHOUSE_Q424, AI_HACKATHON.WAREHOUSE_Q125, AI_HACKATHON.WAREHOUSE_Q225, AI_HACKATHON.WAREHOUSE_Q325
        - Material (NVARCHAR(50), PRIMARY KEY): Material number (SECONDARY KEY referencing WAREHOUSE.Artikelnummer)
        - Materialbezeichnung (NVARCHAR(500)): Material description
        - Datum_von (DATE): Start date of the entry
        - Datum_bis (DATE): End date of the entry
        - AnfangsbestandStück (DOUBLE): Starting stock in pieces
        - Zugang_Stück (DOUBLE): Stock received in pieces
        - Abgang_Stück (DOUBLE): Stock issued in pieces
        - EndbestandStück (DOUBLE): Ending stock in pieces (calculated as Anfangsbestand + Zugang - Abgang)
        - durchschnittlicher_Lagerbestand_Stück (DOUBLE): Average stock in pieces (calculated as (Anfangsbestand + Endbestand) / 2)
        - LUH (DOUBLE): Stock used for leave in pieces
        - durchschnittlicher_Lagerdauer_in_Tagen (DOUBLE): Average stock duration in days
        - Bedarfsprognosegemäß_APO_quartalsweise (DOUBLE): Demand forecast from APO in pieces
        - Lagerreichweite_Anzahl_Quartale (DOUBLE): Stock coverage in number of quarters
        - Lagerreichweite_Anzahl_Jahre (DOUBLE): Stock coverage in number of years

        CRITICAL SQL RULES:
        1. UNION ALL Usage (IMPORTANT):
           - DO: Use UNION ALL INSIDE subqueries/CTEs to combine quarterly tables:
             SELECT * FROM (
               SELECT * FROM "AI_HACKATHON"."WAREHOUSE_Q424"
               UNION ALL SELECT * FROM "AI_HACKATHON"."WAREHOUSE_Q125"
               UNION ALL SELECT * FROM "AI_HACKATHON"."WAREHOUSE_Q225"
               UNION ALL SELECT * FROM "AI_HACKATHON"."WAREHOUSE_Q325"
             ) AS combined_quarters
           - DON'T: Use UNION ALL to combine separate queries with LIMIT (causes syntax errors):
             SELECT ... LIMIT 1 UNION ALL SELECT ... LIMIT 1  -- INVALID!
           - For highest AND lowest values, use CTEs or subqueries in SELECT clause

        2. NO Parameter Placeholders:
           - NEVER use :variable, :artikelnummer, or any bind parameters
           - Use literal values directly in WHERE clauses

        3. NO Placeholder Strings:
           - NEVER use '<Materialnummer>', '<Bitte_eingeben>', or similar placeholders
           - If a material number is in the question, use it directly; otherwise query all materials

        4. Column Aliases in CTEs:
           - Avoid umlauts (ä, ö, ü, ß) in CTE column aliases as they can cause syntax errors
           - Use: AS turnover_rate, AS umschlag_rate, AS lager_rate
           - Avoid in CTEs: AS Lagerumschlagshäufigkeit (can cause errors)
           - OK in final SELECT: AS "Lagerumschlagshäufigkeit"

        5. Pattern for Highest AND Lowest Queries:
           SAP HANA requires separate CTEs - DO NOT use scalar subqueries in SELECT:
           WITH turnover AS (
             SELECT "Material", "Materialbezeichnung",
               SUM("Abgang_Stück") / NULLIF(AVG("durchschnittlicher_Lagerbestand_Stück"), 0) AS rate
             FROM (...quarterly UNION ALL...) AS q
             GROUP BY "Material", "Materialbezeichnung"
           ),
           highest AS (SELECT "Material", "Materialbezeichnung", rate FROM turnover ORDER BY rate DESC LIMIT 1),
           lowest AS (SELECT "Material", "Materialbezeichnung", rate FROM turnover ORDER BY rate ASC LIMIT 1)
           SELECT h."Material" AS highest_material, h."Materialbezeichnung" AS highest_description, h.rate AS highest_rate,
                  l."Material" AS lowest_material, l."Materialbezeichnung" AS lowest_description, l.rate AS lowest_rate
           FROM highest h, lowest l

        6. NO Semicolons:
           - NEVER end SQL with semicolon (;) - causes SAP HANA syntax errors
           - End SQL statements without any terminator

        7. General:
           - Join tables: WAREHOUSE.Artikelnummer = WAREHOUSE_Qxxx.Material
           - Always use double quotes for schema/table/column names
           - Use LIMIT to restrict results
           - Use NULLIF to avoid division by zero
        """
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[GERMAN_AGENT] Attempt {attempt}/{max_retries}")
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": schema_context}]},
                {"role": "user", "content": [{"type": "text", "text": f"Generate SQL for: {user_query}\nReturn ONLY the SQL statement. If this requires inventory data (quantities, storage locations, etc.), inform the user to use the inventory tool instead."}]}
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
            
            logger.info(f"[GERMAN_AGENT] Generated SQL: {sql_statement}")
            
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
            
            logger.info(f"[GERMAN_AGENT] Query executed successfully. Rows returned: {len(results)}")
            logger.info(f"[GERMAN_AGENT] Column names: {column_names}")
            
            cursor.close()
            conn.close()
            
            # Check if results are null or empty
            if results is None or len(results) == 0:
                logger.warning(f"[GERMAN_AGENT] Attempt {attempt}/{max_retries}: Query returned no results. Retrying...")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"[GERMAN_AGENT] All {max_retries} attempts returned no results")
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
            
            logger.info(f"[GERMAN_AGENT] Formatted {len(formatted_results)} results on attempt {attempt}")
            
            return {
                "query": user_query,
                "sql_generated": sql_statement,
                "row_count": len(results),
                "results": formatted_results,
                "note": "Results limited to 50 rows" if len(results) > 50 else "All results shown",
                "attempts": attempt
            }
            
        except Exception as e:
            logger.error(f"[GERMAN_AGENT] Attempt {attempt}/{max_retries} - Error executing database query: {str(e)}")
            if attempt < max_retries:
                logger.info(f"[GERMAN_AGENT] Retrying due to error...")
                continue
            else:
                logger.error(f"[GERMAN_AGENT] All {max_retries} attempts failed")
                return {"error": f"Error executing database query after {max_retries} attempts: {str(e)}"}
    
def final_prompt_warehouse(orig_result):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a warehouse management specialist focusing the quarterly management of warehouse items. You have previously asked the following question: '{orig_result['query']}'" +
                    f" The results from the database query are as follows: '{orig_result['results']}'. Based on these results, provide a concise and informative answer to the user's original question." +
                    "If asked in German, answer is German."
                }
            ]
        }
    ]
    return messages

def final_result(prompt, model_name="gpt-4o", stream=False):
    logger.info(f"[GERMAN_AGENT] Processing final result with model: {model_name}, stream={stream}")
    result = query_database_with_sql_tool(prompt)
    
    if "error" in result:
        logger.error(f"[GERMAN_AGENT] Database query failed: {result['error']}")
        return {
            "response": f"Database query failed: {result['error']}",
            "sql_query": None,
            "error": True
        }
    
    logger.info(f"[GERMAN_AGENT] Generating final response from {result.get('row_count', 0)} database results")
    final_messages = final_prompt_warehouse(result)
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
        logger.info(f"[GERMAN_AGENT] Final response generated successfully (length: {len(response_text)} chars)")
        
        try:
            return {
                "response": response_text,
                "sql_query": result.get("sql_generated"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])
            }
        except Exception as e:
            logger.error(f"[GERMAN_AGENT] Error processing final response: {str(e)}")
            return {
                "response": f"Error processing final response: {str(e)}",
                "sql_query": result.get("sql_generated"),
                "results": result.get("results", []),
                "error": True
            }