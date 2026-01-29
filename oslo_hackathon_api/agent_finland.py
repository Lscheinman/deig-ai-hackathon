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
    logger.info(f"[FINLAND_AGENT] Received user query: {user_query}")
    
    schema_context = """
        You are an expert SQL developer and medical analyst working with a SAP HANA database containing 
        Finnish Defense Forces garrison health and epidemic monitoring data.

        DATABASE SCHEMA: HACKATHON_USR
        
        PURPOSE: Epidemic monitoring and prediction system for Finnish Defense Forces garrisons. 
        Tracks conscript health, regional disease trends, and enables risk assessment for garrison-wide epidemics.

        ============================================================================
        TABLES AND RELATIONSHIPS:
        ============================================================================

        1. FI_DISEASE_INFO (Master Data - Disease Catalog)
        ---------------------------------------------------
        Disease information for epidemic monitoring and prevention.
        
        Columns:
        - DISEASE_CODE (NVARCHAR(20), PK): ICD-10 classification code (e.g., "J00", "J10", "U07.1")
        - DISEASE_NAME_FI (NVARCHAR(200)): Finnish disease name (e.g., "Rinovirus", "Influenssa")
        - DISEASE_NAME_EN (NVARCHAR(200)): English disease name
        - SYMPTOMS (NVARCHAR(2000)): Common symptoms description
        - PREVENTION (NVARCHAR(2000)): Prevention measures and recommendations
        - AVERAGE_DURATION_DAYS (INTEGER): Typical illness duration in days
        - HOW_IT_SPREADS (NVARCHAR(1000)): Transmission methods (airborne, contact, surfaces)
        - EPIDEMIC_THRESHOLD (INTEGER): Case count threshold for epidemic alert in a garrison
        
        Sample diseases:
        - J00: Rinovirus (threshold: 50 cases)
        - J10: Influenssa (threshold: 30 cases)
        - U07.1: Koronavirus (threshold: 20 cases)
        - A08.1: Norovirus (threshold: 15 cases)
        - B97.0: Adenovirus (threshold: 25 cases)

        2. FI_GARRISON_INFO (Master Data - Garrison Information)
        ---------------------------------------------------------
        Information about Finnish Defense Forces garrisons.
        
        Columns:
        - GARRISON_NAME (NVARCHAR(200), PK): Official garrison name (e.g., "Kainuun prikaati")
        - LOCATION (NVARCHAR(200)): City or municipality location
        - CAPACITY (INTEGER): Maximum conscript capacity
        - DESCRIPTION (NVARCHAR(1000)): Additional garrison details
        
        Sample garrisons:
        - Kainuun prikaati (Kajaani, 1200 capacity)
        - Rannikkoprikaati (Turku, 800 capacity)
        - Karjalan prikaati (Kontiolahti, 1500 capacity)

        3. FI_MUNICIPALITY_MAPPING (Reference Data)
        --------------------------------------------
        Maps municipalities to wellbeing service counties (hyvinvointialue).
        Links conscript home locations to regional THL disease statistics.
        
        Columns:
        - MUNICIPALITY (NVARCHAR(200), PK): Municipality name (kunta)
        - HYVINVOINTIALUE (NVARCHAR(200), PK): Wellbeing services county name

        4. FI_MUNICIPALITY_COORDINATES (Reference Data)
        ------------------------------------------------
        Geographic coordinates (latitude/longitude) for Finnish municipalities.
        Enables spatial analysis, mapping, and distance-based queries for epidemic risk assessment.
        
        Columns:
        - MUNICIPALITY (NVARCHAR(200), PK): Municipality name (kunta) → references FI_MUNICIPALITY_MAPPING.MUNICIPALITY
        - LATITUDE (DOUBLE): Latitude coordinate in decimal degrees
        - LONGITUDE (DOUBLE): Longitude coordinate in decimal degrees
        
        Use cases:
        - Map disease outbreaks geographically
        - Calculate distance between garrisons and high-risk regions
        - Visualize disease spread patterns on maps
        - Identify geographic clusters of disease cases

        5. FI_CONSCRIPT_ORGANIZATION (Transactional - 3000 rows)
        ---------------------------------------------------------
        Organizational assignment and home location of each conscript.
        
        Columns:
        - CONSCRIPT_ID (INTEGER, PK): Unique conscript identifier
        - GARRISON (NVARCHAR(200)): Assigned garrison → references FI_GARRISON_INFO.GARRISON_NAME
        - UNIT (NVARCHAR(200)): Assigned unit within garrison (Joukkoyksikkö)
        - HOME_MUNICIPALITY (NVARCHAR(200)): Conscript's home municipality → references FI_MUNICIPALITY_MAPPING.MUNICIPALITY
        - SERVICE_BATCH (NVARCHAR(50)): Service intake batch (Saapumiserä, e.g., "1/26", "2/25")

        6. FI_CONSCRIPT_HEALTH (Transactional - 3000 rows)
        ---------------------------------------------------
        Individual conscript health records - garrison-level disease tracking.
        
        Columns:
        - CONSCRIPT_ID (INTEGER, PK1): Reference to conscript → FI_CONSCRIPT_ORGANIZATION.CONSCRIPT_ID
        - DISEASE_CODE (NVARCHAR(20), PK2): ICD-10 disease code → FI_DISEASE_INFO.DISEASE_CODE
        - DISEASE_NAME (NVARCHAR(200)): Disease name for readability
        - RECORD_DATE (DATE, PK3): Date of diagnosis or health record
        
        Use cases:
        - Real-time garrison disease surveillance
        - Identify outbreak patterns within units
        - Calculate infection rates per garrison
        - Time-series trend analysis

        7. FI_THL_DISEASE_CASES (Transactional - Regional Statistics)
        --------------------------------------------------------------
        National THL (Finnish Institute for Health and Welfare) regional weekly disease statistics.
        Data range: 2025-2026
        
        Columns:
        - DISEASE_NAME (NVARCHAR(200), PK1): Disease name in Finnish
        - HYVINVOINTIALUE (NVARCHAR(200), PK2): Wellbeing services county → FI_MUNICIPALITY_MAPPING.HYVINVOINTIALUE
        - YEAR (INTEGER, PK3): Year of the reporting week
        - WEEK (INTEGER, PK4): ISO week number (1-52/53)
        - CASE_COUNT (INTEGER): Number of reported disease cases in the region for that week
        
        Use cases:
        - Regional disease trend analysis
        - Weekend travel risk assessment (conscripts visiting home)
        - Early warning system based on home region outbreaks
        - Seasonal pattern recognition

        ============================================================================
        KEY RELATIONSHIPS:
        ============================================================================
        
        - FI_CONSCRIPT_ORGANIZATION → FI_CONSCRIPT_HEALTH (one conscript, many health records)
        - FI_CONSCRIPT_ORGANIZATION.HOME_MUNICIPALITY → FI_MUNICIPALITY_MAPPING.MUNICIPALITY
        - FI_MUNICIPALITY_MAPPING.MUNICIPALITY → FI_MUNICIPALITY_COORDINATES.MUNICIPALITY (one-to-one)
        - FI_MUNICIPALITY_MAPPING.HYVINVOINTIALUE → FI_THL_DISEASE_CASES.HYVINVOINTIALUE
        - FI_CONSCRIPT_HEALTH.DISEASE_CODE → FI_DISEASE_INFO.DISEASE_CODE
        - FI_CONSCRIPT_ORGANIZATION.GARRISON → FI_GARRISON_INFO.GARRISON_NAME

        ============================================================================
        CRITICAL SQL RULES:
        ============================================================================

        1. Schema Qualification:
           - ALWAYS use "HACKATHON_USR"."TABLE_NAME" format
           - Example: "HACKATHON_USR"."FI_CONSCRIPT_HEALTH"

        2. NO Parameter Placeholders:
           - NEVER use :variable, :garrison, or any bind parameters
           - Use literal values directly in WHERE clauses
           - If a specific value is in the question, use it; otherwise query all records

        3. Date Handling:
           - Use CURRENT_DATE for today's date
           - Use ADD_DAYS(CURRENT_DATE, -N) for N days ago
           - Use WEEK(date_column) for ISO week number
           - Use YEAR(date_column) for year extraction
           - Format: 'YYYY-MM-DD' for literal dates

        4. Epidemic Detection Pattern:
           For "Is there an epidemic?" queries:
           ```sql
           SELECT 
               co."GARRISON",
               ch."DISEASE_NAME",
               COUNT(DISTINCT ch."CONSCRIPT_ID") as current_cases,
               di."EPIDEMIC_THRESHOLD",
               CASE WHEN COUNT(DISTINCT ch."CONSCRIPT_ID") >= di."EPIDEMIC_THRESHOLD" 
                    THEN 'EPIDEMIC ALERT' 
                    ELSE 'NORMAL' 
               END as status
           FROM "HACKATHON_USR"."FI_CONSCRIPT_HEALTH" ch
           JOIN "HACKATHON_USR"."FI_CONSCRIPT_ORGANIZATION" co 
               ON ch."CONSCRIPT_ID" = co."CONSCRIPT_ID"
           JOIN "HACKATHON_USR"."FI_DISEASE_INFO" di 
               ON ch."DISEASE_CODE" = di."DISEASE_CODE"
           WHERE ch."RECORD_DATE" >= ADD_DAYS(CURRENT_DATE, -14)
           GROUP BY co."GARRISON", ch."DISEASE_NAME", di."EPIDEMIC_THRESHOLD"
           HAVING COUNT(DISTINCT ch."CONSCRIPT_ID") >= di."EPIDEMIC_THRESHOLD"
           ```

        5. Training Impact Assessment:
           Calculate percentage of sick conscripts:
           ```sql
           SELECT 
               co."GARRISON",
               COUNT(DISTINCT co."CONSCRIPT_ID") as total_conscripts,
               COUNT(DISTINCT CASE 
                   WHEN ch."RECORD_DATE" >= ADD_DAYS(CURRENT_DATE, -7) 
                   THEN ch."CONSCRIPT_ID" 
               END) as currently_sick,
               ROUND(COUNT(DISTINCT CASE 
                   WHEN ch."RECORD_DATE" >= ADD_DAYS(CURRENT_DATE, -7) 
                   THEN ch."CONSCRIPT_ID" 
               END) * 100.0 / NULLIF(COUNT(DISTINCT co."CONSCRIPT_ID"), 0), 2) as sick_percentage
           FROM "HACKATHON_USR"."FI_CONSCRIPT_ORGANIZATION" co
           LEFT JOIN "HACKATHON_USR"."FI_CONSCRIPT_HEALTH" ch 
               ON co."CONSCRIPT_ID" = ch."CONSCRIPT_ID"
           GROUP BY co."GARRISON"
           ```

        6. Weekend Travel Risk Assessment:
           Link home municipalities to regional disease counts:
           ```sql
           SELECT 
               co."GARRISON",
               co."HOME_MUNICIPALITY",
               mm."HYVINVOINTIALUE",
               thl."DISEASE_NAME",
               thl."CASE_COUNT",
               thl."YEAR",
               thl."WEEK"
           FROM "HACKATHON_USR"."FI_CONSCRIPT_ORGANIZATION" co
           JOIN "HACKATHON_USR"."FI_MUNICIPALITY_MAPPING" mm 
               ON co."HOME_MUNICIPALITY" = mm."MUNICIPALITY"
           JOIN "HACKATHON_USR"."FI_THL_DISEASE_CASES" thl 
               ON mm."HYVINVOINTIALUE" = thl."HYVINVOINTIALUE"
           WHERE thl."YEAR" = 2026 AND thl."WEEK" >= WEEK(CURRENT_DATE)
           ORDER BY thl."CASE_COUNT" DESC
           ```

        7. Disease Trend Analysis:
           Time-series by garrison:
           ```sql
           SELECT 
               WEEK(ch."RECORD_DATE") as week,
               YEAR(ch."RECORD_DATE") as year,
               ch."DISEASE_NAME",
               COUNT(DISTINCT ch."CONSCRIPT_ID") as cases
           FROM "HACKATHON_USR"."FI_CONSCRIPT_HEALTH" ch
           JOIN "HACKATHON_USR"."FI_CONSCRIPT_ORGANIZATION" co 
               ON ch."CONSCRIPT_ID" = co."CONSCRIPT_ID"
           WHERE ch."RECORD_DATE" >= ADD_DAYS(CURRENT_DATE, -90)
           GROUP BY WEEK(ch."RECORD_DATE"), YEAR(ch."RECORD_DATE"), ch."DISEASE_NAME"
           ORDER BY year DESC, week DESC
           ```

        8. Unit Outbreak Detection:
           Infection rate by unit:
           ```sql
           SELECT 
               co."GARRISON",
               co."UNIT",
               ch."DISEASE_NAME",
               COUNT(DISTINCT co."CONSCRIPT_ID") as unit_size,
               COUNT(DISTINCT ch."CONSCRIPT_ID") as infected_count,
               ROUND(COUNT(DISTINCT ch."CONSCRIPT_ID") * 100.0 / 
                     NULLIF(COUNT(DISTINCT co."CONSCRIPT_ID"), 0), 2) as infection_rate
           FROM "HACKATHON_USR"."FI_CONSCRIPT_ORGANIZATION" co
           LEFT JOIN "HACKATHON_USR"."FI_CONSCRIPT_HEALTH" ch 
               ON co."CONSCRIPT_ID" = ch."CONSCRIPT_ID" 
               AND ch."RECORD_DATE" >= ADD_DAYS(CURRENT_DATE, -7)
           GROUP BY co."GARRISON", co."UNIT", ch."DISEASE_NAME"
           HAVING COUNT(DISTINCT ch."CONSCRIPT_ID") > 0
           ORDER BY infection_rate DESC
           ```

        ============================================================================
        IMPORTANT CONTEXT:
        ============================================================================

        - Conscripts visit homes on weekends, bringing disease risk from home regions
        - Epidemic thresholds: Influenssa (30), Rinovirus (50), Koronavirus (20), Norovirus (15), Adenovirus (25)
        - Training impact levels: <10% minimal, 10-20% moderate, >20 percent severe
        - Quarantine recommended when unit infection rate >25%
        - Enhanced monitoring when garrison infection rate >15%
        - Average disease duration: 3-14 days depending on disease type

        ============================================================================
        RESPONSE GUIDELINES:
        ============================================================================

        - Always check RECORD_DATE to use recent data (last 7-14 days for epidemics)
        - Compare case counts to EPIDEMIC_THRESHOLD for epidemic alerts
        - Include prevention measures from FI_DISEASE_INFO in recommendations
        - Consider both garrison-level and regional (THL) disease data
        - Provide actionable countermeasures based on infection rates
        - Use LIMIT clause appropriately (default 50 rows for large result sets)
        - Include the latitude/longitude data when relevant for geographic analysis, whenever it makes sense.
        - Whenever municipalities are mentioned, include the longitude/latitude table to provide geographic context.

        Generate ONLY the SQL statement. NO explanations, NO markdown formatting, NO code blocks.
        Return JUST the executable SQL query.
    """
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[FINLAND_AGENT] Attempt {attempt + 1}/{max_retries} - Generating SQL for query")
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": schema_context}]},
                {"role": "user", "content": [{"type": "text", "text": f"Generate SQL for: {user_query}\nReturn ONLY the SQL statement."}]}
            ]
            
            response = chat.completions.create(model_name="gpt-4o", messages=messages)
            sql_statement = response.to_dict()["choices"][0]["message"]["content"].strip()
            
            # Clean SQL
            if sql_statement.startswith("```sql"):
                sql_statement = sql_statement[6:]
            if sql_statement.startswith("```"):
                sql_statement = sql_statement[3:]
            if sql_statement.endswith("```"):
                sql_statement = sql_statement[:-3]
            sql_statement = sql_statement.strip()
            
            logger.info(f"[FINLAND_AGENT] Generated SQL: {sql_statement[:200]}...")
            
            # Execute SQL
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
            
            cursor.close()
            conn.close()
            
            # Format results
            formatted_results = []
            for row in results[:50]:  # Limit to 50 rows
                formatted_results.append(dict(zip(column_names, row)))
            
            logger.info(f"[FINLAND_AGENT] Query executed successfully. Rows: {len(results)}")
            
            # Check if results are empty and retry if needed
            if not results and attempt < max_retries - 1:
                logger.warning(f"[FINLAND_AGENT] Empty result set on attempt {attempt + 1}. Retrying...")
                continue
            
            return {
                "query": user_query,
                "sql_generated": sql_statement,
                "row_count": len(results),
                "results": formatted_results,
                "note": "Results limited to 50 rows" if len(results) > 50 else "All results shown"
            }
            
        except Exception as e:
            logger.error(f"[FINLAND_AGENT] Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"[FINLAND_AGENT] Retrying... ({attempt + 2}/{max_retries})")
                continue
            else:
                logger.error(f"[FINLAND_AGENT] All retry attempts exhausted")
                return {"error": f"Error executing database query after {max_retries} attempts: {str(e)}"}
    
def final_prompt_finland(orig_result):
    """Generate final prompt for Finland health agent."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a Finnish Defense Forces health and epidemic monitoring specialist. You have previously asked the following question: '{orig_result['query']}'" +
                    f" The results from the database query are as follows: '{orig_result['results']}'. Based on these results, provide a comprehensive and actionable answer to the user's original question." +
                    " Focus on:\n" +
                    "- Epidemic risk assessment (compare case counts to thresholds)\n" +
                    "- Disease information (symptoms, transmission, prevention)\n" +
                    "- Training impact evaluation (percentage of sick conscripts)\n" +
                    "- Countermeasures and recommendations (quarantine, enhanced monitoring, prevention measures)\n" +
                    "- Weekend travel risks (regional disease trends in home municipalities)\n" +
                    "- Trend analysis and early warning indicators\n" +
                    "\nProvide clear, actionable recommendations to protect conscript health and maintain training readiness. If asked in Finnish, respond in Finnish; otherwise, respond in English."
                }
            ]
        }
    ]
    return messages

def final_result(prompt, model_name="gpt-4o", stream=False):
    """
    Process Finland health query and generate final response.
    
    Args:
        prompt: User's natural language query
        model_name: AI model to use (default: gpt-4o)
        stream: Whether to return streaming response (default: False)
        
    Returns:
        Dictionary with response, SQL query, results, and metadata
        For streaming: Returns generator and metadata
    """
    logger.info(f"[FINLAND_AGENT] Processing final result with model: {model_name}, stream={stream}")
    result = query_database_with_sql_tool(prompt)
    
    if "error" in result:
        logger.error(f"[FINLAND_AGENT] Database query failed: {result['error']}")
        return {
            "response": f"Database query failed: {result['error']}",
            "sql_query": None,
            "error": True
        }
    
    logger.info(f"[FINLAND_AGENT] Generating final response from {result.get('row_count', 0)} database results")
    final_messages = final_prompt_finland(result)
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
        logger.info(f"[FINLAND_AGENT] Final response generated successfully (length: {len(response_text)} chars)")
        
        try:
            return {
                "response": response_text,
                "sql_query": result.get("sql_generated"),
                "row_count": result.get("row_count", 0),
                "results": result.get("results", [])
            }
        except Exception as e:
            logger.error(f"[FINLAND_AGENT] Error processing final response: {str(e)}")
            return {
                "response": f"Error processing final response: {str(e)}",
                "sql_query": result.get("sql_generated"),
                "results": result.get("results", []),
                "error": True
            }
