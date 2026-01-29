"""
Explosives Intelligence Agent Module
====================================
This module provides AI-powered tools for querying and analyzing explosives data
from a SAP HANA database, including compatibility checks, inventory queries, and
natural language SQL generation.
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, Any, List
from dotenv import load_dotenv
from hdbcli import dbapi

# Load environment variables
load_dotenv()

# HANA Credentials
HANA_HOST = os.getenv("HANA_HOST")
HANA_PORT = os.getenv("HANA_PORT")
HANA_USER = os.getenv("HANA_USER")
HANA_PASSWORD = os.getenv("HANA_PASSWORD")

# S/4 Credentials
S4_USER = os.getenv("S4_USER")
S4_PASSWORD = os.getenv("S4_PASSWORD")

# AI Core Credentials
os.environ['AICORE_CLIENT_ID'] = os.getenv("AICORE_CLIENT_ID")
os.environ['AICORE_CLIENT_SECRET'] = os.getenv("AICORE_CLIENT_SECRET")
os.environ['AICORE_AUTH_URL'] = os.getenv("AICORE_AUTH_URL")
os.environ['AICORE_BASE_URL'] = os.getenv("AICORE_BASE_URL")
os.environ['AICORE_RESOURCE_GROUP'] = os.getenv("AICORE_RESOURCE_GROUP")

from gen_ai_hub.proxy.native.openai import chat

# ============================================================================
# HELPER FUNCTION - Fetch Inventory from S/4
# ============================================================================

def fetch_inventory_from_s4():
    """
    Fetch current inventory data from S/4 HANA and join with HANA master data.
    Returns a DataFrame with complete inventory information.
    """
    try:
        # Connect to HANA for master data
        conn = dbapi.connect(address=HANA_HOST, port=HANA_PORT, user=HANA_USER, password=HANA_PASSWORD)
        cursor = conn.cursor()
        
        # Get HMSD data
        cursor.execute('SELECT * FROM "EXPLOSIVES"."HMSD"')
        result = cursor.fetchall()
        df_hmsd = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
        
        # Get SLOC data
        cursor.execute('SELECT * FROM "EXPLOSIVES"."SLOC"')
        result = cursor.fetchall()
        df_sloc = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
        
        cursor.close()
        conn.close()
        
        # Fetch from S/4 with filter for non-zero quantities and specific storage locations
        filter_condition = "(MatlWrhsStkQtyInMatlBaseUnit%20gt%200)%20and%20(StorageLocation%20eq%20%2700AJ%27%20or%20StorageLocation%20eq%20%2700AK%27%20or%20StorageLocation%20eq%20%2700AL%27%20or%20StorageLocation%20eq%20%2700AM%27%20or%20StorageLocation%20eq%20%2700AN%27%20or%20StorageLocation%20eq%20%2700AO%27%20or%20StorageLocation%20eq%20%2700AP%27%20or%20StorageLocation%20eq%20%2700AQ%27%20or%20StorageLocation%20eq%20%2700AR%27)"
        url = f"https://coeportal515.saphosting.de/sap/opu/odata/sap/API_MATERIAL_STOCK_SRV/A_MatlStkInAcctMod?$format=json&sap-client=600&$filter={filter_condition}"
        
        response = requests.get(url, auth=HTTPBasicAuth(S4_USER, S4_PASSWORD), timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            entity_data = data.get('d', {}).get('results', [])
            df_inventory = pd.DataFrame(entity_data)
        else:
            df_inventory = pd.DataFrame()
        
        if df_inventory.empty:
            return pd.DataFrame()
        
        # Filter relevant columns
        df_inventory = df_inventory[['Material', 'StorageLocation', 'Batch', 'MaterialBaseUnit', 'MatlWrhsStkQtyInMatlBaseUnit']]
        
        # Convert types
        df_inventory["Material"] = df_inventory["Material"].apply(lambda x: int(x))
        df_inventory["MatlWrhsStkQtyInMatlBaseUnit"] = df_inventory["MatlWrhsStkQtyInMatlBaseUnit"].apply(lambda x: int(x))
        
        # Join with HMSD
        df_inventory = df_inventory.merge(
            df_hmsd[["Material", "Material_Description", "HD", "CG", "NEW_KG"]], 
            on="Material", 
            how="left"
        )
        
        # Join with SLOC
        df_inventory = df_inventory.merge(
            df_sloc[["SLOC", "SLOC_Description"]], 
            left_on="StorageLocation", 
            right_on="SLOC", 
            how="left"
        )
        
        # Select and rename columns
        df_inventory = df_inventory[[
            "Material", "Material_Description", "HD", "CG", "NEW_KG", 
            "StorageLocation", "SLOC_Description", "Batch", "MaterialBaseUnit", 
            "MatlWrhsStkQtyInMatlBaseUnit"
        ]]
        
        # Calculate total NEW
        df_inventory["NEW_Total_KG"] = df_inventory["NEW_KG"] * df_inventory["MatlWrhsStkQtyInMatlBaseUnit"]
        
        # Rename columns for consistency
        df_inventory.columns = [
            "Material", "Description", "Hazard_Division", "CG", "NEW_EA_KG",
            "Storage_Location", "SLOC_Desc", "Batch", "MaterialBaseUnit",
            "Quantity", "NEW_Total_KG"
        ]
        
        return df_inventory
        
    except Exception as e:
        print(f"Error fetching inventory from S/4: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# IATG NEW AGGREGATION CLASS
# ============================================================================

class IATGAggregator:
    """
    IATG (International Ammunition Technical Guidelines) NEW Aggregation.
    
    Implements IATG rules for aggregating Net Explosive Weight (NEW) when multiple
    Hazard Divisions (HD) are stored together in the same location. This is used
    for calculating Quantity Distance (QTD) requirements.
    """
    
    def __init__(self, inventory):
        """
        Args:
            inventory: dict of HD: weight (e.g., {"1.2": 50, "1.3.1": 100})
        """
        self.raw_inv = inventory
        # Standardizing all variants into their primary IATG aggregation buckets
        self.weights = {
            "1.1": inventory.get("1.1", 0) + inventory.get("1.5", 0),
            # IATG Rule: HD 1.2 (unspecified) and 1.2.3 are treated as 1.2.1
            "1.2.1": (inventory.get("1.2.1", 0) + 
                      inventory.get("1.2.3", 0) + 
                      inventory.get("1.2", 0)),
            "1.2.2": inventory.get("1.2.2", 0),
            "1.3": (inventory.get("1.3", 0) + 
                    inventory.get("1.3.1", 0) + 
                    inventory.get("1.3.2", 0)),
            "1.4": inventory.get("1.4", 0),
            "1.6": inventory.get("1.6", 0)
        }

    def calculate_aggregation(self, verbose=False):
        """
        Calculate the aggregated HD and NEW according to IATG rules.
        
        Args:
            verbose: If True, print step-by-step trace
            
        Returns:
            tuple: (aggregated_hd, aggregated_new_kg)
        """
        if verbose:
            print("--- IATG Step-by-Step Aggregation Trace ---")
            for hd, weight in self.raw_inv.items():
                if weight > 0:
                    print(f"Input: {hd} = {weight} kg")
            print("-" * 40)

        # 1. The 1.1/1.5 Rule - Aggregates everything EXCEPT 1.4
        if self.weights["1.1"] > 0:
            total = sum(self.weights.values()) - self.weights["1.4"]
            if verbose:
                print(f"[RULE 1] HD 1.1/1.5 detected.")
                print(f"         Aggregating ALL items to 1.1 except 1.4 which is ignored: {total} kg")
                if self.weights["1.4"] > 0:
                    print(f"         (HD 1.4 of {self.weights['1.4']} kg is ignored per IATG)")
            return "1.1", total

        # 2. The 1.2.1 Rule (Includes general 1.2 and 1.2.3)
        if self.weights["1.2.1"] > 0:
            total = (self.weights["1.2.1"] + self.weights["1.2.2"] + 
                     self.weights["1.3"] + self.weights["1.6"])
            if verbose:
                print(f"[RULE 2] HD 1.2.1 (or generic 1.2) detected.")
                print(f"         Aggregating 1.2.x, 1.3, and 1.6 as 1.2.1: {total} kg")
                if self.weights["1.4"] > 0:
                    print(f"         (HD 1.4 of {self.weights['1.4']} kg is ignored per IATG)")            
            return "1.2.1", total

        # 3. The 1.2.2 Rule
        if self.weights["1.2.2"] > 0:
            total = self.weights["1.2.2"] + self.weights["1.3"] + self.weights["1.6"]
            if verbose:
                print(f"[RULE 3] HD 1.2.2 detected.")
                print(f"         Aggregating 1.2.2, 1.3, and 1.6 as 1.2.2: {total} kg")
                if self.weights["1.4"] > 0:
                    print(f"         (HD 1.4 of {self.weights['1.4']} kg is ignored per IATG)")
            return "1.2.2", total

        # 4. The 1.3 Rule
        if self.weights["1.3"] > 0:
            total = self.weights["1.3"] + self.weights["1.6"]
            if verbose:
                print(f"[RULE 4] HD 1.3 detected.")
                print(f"         Aggregating 1.3 and 1.6 as 1.3: {total} kg")
                if self.weights["1.4"] > 0:
                    print(f"         (HD 1.4 of {self.weights['1.4']} kg is ignored per IATG)")
            return "1.3", total

        # 5. The 1.6 Rule
        if self.weights["1.6"] > 0:
            if verbose:
                print(f"[RULE 5] Only HD 1.6 and 1.4 present. 1.4 is ignored")
                if self.weights["1.4"] > 0:
                    print(f"         (HD 1.4 of {self.weights['1.4']} kg is ignored per IATG)")
            return "1.6", self.weights["1.6"]

        # 6. The 1.4 Rule
        if verbose:
            print(f"[RULE 6] Only HD 1.4")
        return "1.4", self.weights["1.4"]

# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def get_compatibility_info_tool(material_number: int) -> Dict[str, Any]:
    """
    Get comprehensive compatibility information for a specific explosive material.
    
    Args:
        material_number: The material number to look up
        
    Returns:
        Dictionary containing compatibility group, descriptions, and matrix
    """
    try:
        conn = dbapi.connect(
            address=HANA_HOST,
            port=HANA_PORT,
            user=HANA_USER,
            password=HANA_PASSWORD
        )
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT "EXPLOSIVES"."HMSD"."CG", 
                   "EXPLOSIVES"."CG"."CG_Description",
                   "EXPLOSIVES"."HMSD"."Material_Description", 
                   "EXPLOSIVES"."HMSD"."HD"
            FROM "EXPLOSIVES"."HMSD" 
            JOIN "EXPLOSIVES"."CG" ON "EXPLOSIVES"."HMSD"."CG" = "EXPLOSIVES"."CG"."CG" 
            WHERE "EXPLOSIVES"."HMSD"."Material" = {material_number} 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not result:
            return {"error": f"Material {material_number} not found in database"}
        
        result_group = result[0]
        result_group_desc = result[1]
        result_material_desc = result[2]
        result_hazard_division = result[3]
        
        # Compatibility matrix
        compatibility_groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'S']
        compatibility_matrix = np.array([
            ["X", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],  # A
            ["0", "X", "1", "1", "1", "1", "1", "0", "0", "0", "0", "0", "X"],  # B
            ["0", "1", "X", "X", "X", "2", "3", "0", "0", "0", "0", "4", "X"],  # C
            ["0", "1", "X", "X", "X", "2", "3", "0", "0", "0", "0", "4", "X"],  # D
            ["0", "1", "X", "X", "X", "2", "3", "0", "0", "0", "0", "4", "X"],  # E
            ["0", "1", "2", "2", "2", "X", "2,3", "0", "0", "0", "0", "0", "X"],  # F
            ["0", "1", "3", "3", "3", "2,3", "X", "0", "0", "0", "0", "0", "X"],  # G
            ["0", "0", "0", "0", "0", "0", "0", "X", "0", "0", "0", "0", "X"],  # H
            ["0", "0", "0", "0", "0", "0", "0", "0", "X", "0", "0", "0", "X"],  # J
            ["0", "0", "0", "0", "0", "0", "0", "0", "0", "X", "0", "0", "0"],  # K
            ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "5", "0", "0"],  # L
            ["0", "0", "4", "4", "4", "0", "0", "0", "0", "0", "0", "6", "7"],  # N
            ["0", "X", "X", "X", "X", "X", "X", "X", "X", "0", "0", "7", "X"],  # S
        ])
        
        comp_index = compatibility_groups.index(result_group)
        compatibility_row = compatibility_matrix[comp_index]
        paired_dict = {cg: str(compatibility_row[i]) for i, cg in enumerate(compatibility_groups)}
        
        return {
            "material_number": material_number,
            "material_description": result_material_desc,
            "compatibility_group": result_group,
            "compatibility_group_description": result_group_desc,
            "hazard_division": result_hazard_division,
            "compatibility_with_other_groups": paired_dict,
            "compatibility_legend": {
                "0": "Not permitted",
                "X": "Permitted",
                "1": "Permitted with warning: CG B fuzes may be stored with articles they'll be assembled with",
                "2": "Permitted with warning: Storage in same building if effectively segregated",
                "3": "Permitted with warning: Mixing CG G is at discretion of National Competent Authority",
                "2,3": "Permitted with warnings for both segregation and CG G mixing",
                "4": "Permitted with warning: CG N stored with C/D/E should be considered as CG D",
                "5": "Permitted with warning: CG L must always be stored separately",
                "6": "Permitted with warning: 1.6N munitions mixing rules apply",
                "7": "Permitted with warning: Mixed 1.6N and 1.4S may be considered as CG N"
            }
        }
    except Exception as e:
        return {"error": f"Error retrieving compatibility info: {str(e)}"}


def query_database_with_sql_tool(user_query: str) -> Dict[str, Any]:
    """
    Generate SQL from natural language query and execute it against the HANA database.
    
    Args:
        user_query: Natural language description of the query
        
    Returns:
        Dictionary containing SQL, results, and metadata
    """
    try:
        schema_context = """
You are an expert SQL developer working with a SAP HANA database named EXPLOSIVES.

The database contains 4 tables with the following schema:

1. Table: EXPLOSIVES.CG (Compatibility Groups)
   - CG (NVARCHAR(50), PRIMARY KEY): Compatibility Group code
   - CG_Description (NVARCHAR(500)): Description

2. Table: EXPLOSIVES.HD (Hazard Divisions)
   - HD (NVARCHAR(50), PRIMARY KEY): Hazard Division code
   - HD_Description (NVARCHAR(500)): Description

3. Table: EXPLOSIVES.SLOC (Storage Locations)
   - SLOC (NVARCHAR(50), PRIMARY KEY): Storage Location code
   - SLOC_Description (NVARCHAR(500)): Description

4. Table: EXPLOSIVES.HMSD (Hazardous Substance Master Data)
   - Material (INTEGER, PRIMARY KEY): Material number
   - Material_Description (NVARCHAR(500)): Description
   - HD (NVARCHAR(50)): Hazard Division
   - CG (NVARCHAR(50)): Compatibility Group
   - NEW_KG (DOUBLE): Net Explosive Weight in KG
   - FSC (INTEGER): Federal Supply Code
   - Interchangeability_Code (NVARCHAR(100))

IMPORTANT NOTES:
- The INVENTORY table is NO LONGER AVAILABLE in HANA. Inventory data is now fetched from S/4 HANA via API.
- For inventory-related queries, you can only query the HMSD table for material master data.
- To get actual inventory quantities and storage locations, use the get_material_inventory tool instead.
- Use double quotes for schema/table/column names.
- Use LIMIT to restrict results.
"""
        
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
        
        # Format results as list of dictionaries
        formatted_results = []
        for row in results[:50]:
            formatted_results.append(dict(zip(column_names, row)))
        
        return {
            "query": user_query,
            "sql_generated": sql_statement,
            "row_count": len(results),
            "results": formatted_results,
            "note": "Results limited to 50 rows" if len(results) > 50 else "All results shown"
        }
        
    except Exception as e:
        return {"error": f"Error executing database query: {str(e)}"}


def get_material_inventory_tool(material_number: int) -> Dict[str, Any]:
    """
    Get detailed inventory information for a specific material.
    Data is fetched live from S/4 HANA system.
    
    Args:
        material_number: The material number to check
        
    Returns:
        Dictionary containing inventory details across all storage locations
    """
    try:
        # Fetch current inventory from S/4
        df_inventory = fetch_inventory_from_s4()
        
        if df_inventory.empty:
            return {"error": "Unable to fetch inventory data from S/4"}
        
        # Filter for specific material
        material_inventory = df_inventory[df_inventory["Material"] == material_number]
        
        if material_inventory.empty:
            return {"error": f"No inventory found for material {material_number}"}
        
        inventory_list = []
        total_quantity = 0
        total_new = 0
        
        for _, row in material_inventory.iterrows():
            inventory_list.append({
                "material": int(row["Material"]),
                "description": row["Description"],
                "hazard_division": row["Hazard_Division"],
                "compatibility_group": row["CG"],
                "new_per_unit_kg": float(row["NEW_EA_KG"]) if pd.notna(row["NEW_EA_KG"]) else 0,
                "storage_location": row["Storage_Location"],
                "storage_location_description": row["SLOC_Desc"],
                "batch": row["Batch"],
                "base_unit": row["MaterialBaseUnit"],
                "quantity": int(row["Quantity"]),
                "total_new_kg": float(row["NEW_Total_KG"]) if pd.notna(row["NEW_Total_KG"]) else 0
            })
            total_quantity += int(row["Quantity"])
            total_new += float(row["NEW_Total_KG"]) if pd.notna(row["NEW_Total_KG"]) else 0
        
        return {
            "material_number": material_number,
            "inventory_locations": inventory_list,
            "total_quantity_all_locations": total_quantity,
            "total_new_all_locations_kg": total_new,
            "number_of_storage_locations": len(inventory_list),
            "data_source": "S/4 HANA (live data)"
        }
        
    except Exception as e:
        return {"error": f"Error retrieving inventory: {str(e)}"}


def list_all_materials_tool() -> Dict[str, Any]:
    """
    List all explosive materials in the database.
    
    Returns:
        Dictionary containing list of all materials with basic information
    """
    try:
        conn = dbapi.connect(
            address=HANA_HOST,
            port=HANA_PORT,
            user=HANA_USER,
            password=HANA_PASSWORD
        )
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT "Material", "Material_Description", "HD", "CG", "NEW_KG"
            FROM "EXPLOSIVES"."HMSD"
            ORDER BY "Material"
            LIMIT 100
        ''')
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        materials_list = []
        for row in results:
            materials_list.append({
                "material_number": row[0],
                "description": row[1],
                "hazard_division": row[2],
                "compatibility_group": row[3],
                "new_kg": row[4]
            })
        
        return {
            "total_materials": len(materials_list),
            "materials": materials_list,
            "note": "Limited to 100 materials for readability. Use get_material_inventory to check current stock levels from S/4."
        }
        
    except Exception as e:
        return {"error": f"Error listing materials: {str(e)}"}


def get_storage_summary_tool() -> Dict[str, Any]:
    """
    Get a summary of all storage locations with IATG-aggregated NEW values.
    Data is fetched live from S/4 HANA system.
    Includes IATG-aggregated NEW values for Quantity Distance calculations.
    
    Returns:
        Dictionary containing storage location summaries with quantities
    """
    try:
        # Fetch current inventory from S/4
        df_inventory = fetch_inventory_from_s4()
        
        if df_inventory.empty:
            return {"error": "Unable to fetch inventory data from S/4"}
        
        # Group by storage location
        storage_summary = df_inventory.groupby(['Storage_Location', 'SLOC_Desc']).agg({
            'Material': 'nunique',
            'Quantity': 'sum',
            'NEW_Total_KG': 'sum'
        }).reset_index()
        
        storage_summary.columns = ['Storage_Location', 'SLOC_Desc', 'Material_Count', 'Total_Items', 'Total_NEW_KG']
        storage_summary = storage_summary.sort_values('Total_NEW_KG', ascending=False)
        
        storage_list = []
        for _, row in storage_summary.iterrows():
            location = row["Storage_Location"]
            
            # Calculate IATG aggregated NEW for this location
            location_inventory = df_inventory[df_inventory["Storage_Location"] == location]
            hd_weights = {}
            for _, inv_row in location_inventory.iterrows():
                hd = inv_row["Hazard_Division"]
                new_kg = float(inv_row["NEW_Total_KG"]) if pd.notna(inv_row["NEW_Total_KG"]) else 0
                if pd.notna(hd) and hd:
                    hd_weights[hd] = hd_weights.get(hd, 0) + new_kg
            
            # Apply IATG aggregation
            aggregated_hd = None
            aggregated_new = 0
            if hd_weights:
                aggregator = IATGAggregator(hd_weights)
                aggregated_hd, aggregated_new = aggregator.calculate_aggregation(verbose=False)
            
            storage_list.append({
                "storage_location": row["Storage_Location"],
                "description": row["SLOC_Desc"],
                "unique_materials": int(row["Material_Count"]),
                "total_items": int(row["Total_Items"]),
                "total_new_kg": float(row["Total_NEW_KG"]) if pd.notna(row["Total_NEW_KG"]) else 0,
                "iatg_aggregated_hd": aggregated_hd,
                "iatg_aggregated_new_kg": round(aggregated_new, 4) if aggregated_new else 0,
                "hazard_divisions_present": list(hd_weights.keys()) if hd_weights else []
            })
        
        return {
            "total_storage_locations": len(storage_list),
            "storage_locations": storage_list,
            "data_source": "S/4 HANA (live data)",
            "note": "IATG aggregated values are provided for Quantity Distance (QTD) calculations"
        }
        
    except Exception as e:
        return {"error": f"Error retrieving storage summary: {str(e)}"}


def calculate_storage_location_qtd_tool(storage_location: str) -> Dict[str, Any]:
    """
    Calculate IATG-aggregated Hazard Division and NEW for a specific storage location.
    This is used for Quantity Distance (QTD) calculations and safety planning.
    
    Args:
        storage_location: Storage location code (e.g., '00AJ')
        
    Returns:
        Dictionary containing QTD calculation details
    """
    try:
        # Fetch current inventory from S/4
        df_inventory = fetch_inventory_from_s4()
        
        if df_inventory.empty:
            return {"error": "Unable to fetch inventory data from S/4"}
        
        # Filter for specific storage location
        location_inventory = df_inventory[df_inventory["Storage_Location"] == storage_location]
        
        if location_inventory.empty:
            return {"error": f"No inventory found for storage location {storage_location}"}
        
        # Get storage location description
        sloc_desc = location_inventory.iloc[0]["SLOC_Desc"] if not location_inventory.empty else "Unknown"
        
        # Group by Hazard Division and sum NEW
        hd_weights = {}
        materials_by_hd = {}
        
        for _, row in location_inventory.iterrows():
            hd = row["Hazard_Division"]
            new_kg = float(row["NEW_Total_KG"]) if pd.notna(row["NEW_Total_KG"]) else 0
            material = int(row["Material"])
            
            if pd.notna(hd) and hd:
                hd_weights[hd] = hd_weights.get(hd, 0) + new_kg
                if hd not in materials_by_hd:
                    materials_by_hd[hd] = []
                materials_by_hd[hd].append({
                    "material": material,
                    "description": row["Description"],
                    "new_kg": new_kg
                })
        
        if not hd_weights:
            return {"error": f"No hazard division data found for storage location {storage_location}"}
        
        # Apply IATG aggregation
        aggregator = IATGAggregator(hd_weights)
        aggregated_hd, aggregated_new = aggregator.calculate_aggregation(verbose=False)
        
        # Prepare detailed breakdown
        hd_breakdown = []
        for hd, weight in hd_weights.items():
            hd_breakdown.append({
                "hazard_division": hd,
                "total_new_kg": round(weight, 4),
                "material_count": len(materials_by_hd.get(hd, [])),
                "materials": materials_by_hd.get(hd, [])
            })
        
        return {
            "storage_location": storage_location,
            "storage_location_description": sloc_desc,
            "hazard_division_breakdown": hd_breakdown,
            "iatg_aggregated_hd": aggregated_hd,
            "iatg_aggregated_new_kg": round(aggregated_new, 4),
            "total_materials": int(location_inventory["Material"].nunique()),
            "total_items": int(location_inventory["Quantity"].sum()),
            "iatg_rule_applied": f"Aggregated to HD {aggregated_hd} per IATG guidelines",
            "data_source": "S/4 HANA (live data)",
            "usage": "Use aggregated values for Quantity Distance (QTD) and safety distance calculations"
        }
        
    except Exception as e:
        return {"error": f"Error calculating QTD for storage location: {str(e)}"}


def calculate_materials_aggregation_tool(material_numbers: List[int]) -> Dict[str, Any]:
    """
    Calculate IATG-aggregated Hazard Division and NEW for a list of specific materials.
    This "what-if" tool helps determine the aggregated HD and NEW if specific materials
    were stored together, useful for planning and safety assessment.
    
    Args:
        material_numbers: List of material numbers to aggregate
        
    Returns:
        Dictionary containing materials aggregation details
    """
    try:
        # Connect to HANA for material master data
        conn = dbapi.connect(address=HANA_HOST, port=HANA_PORT, user=HANA_USER, password=HANA_PASSWORD)
        cursor = conn.cursor()
        
        # Build material list for query
        material_list = ",".join(str(m) for m in material_numbers)
        
        # Get material data including HD and NEW
        cursor.execute(f'''
            SELECT "Material", "Material_Description", "HD", "NEW_KG"
            FROM "EXPLOSIVES"."HMSD"
            WHERE "Material" IN ({material_list})
        ''')
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not results:
            return {"error": f"None of the specified materials found in database"}
        
        # Build HD weights and material details
        hd_weights = {}
        materials_by_hd = {}
        materials_info = []
        
        for row in results:
            material = row[0]
            description = row[1]
            hd = row[2]
            new_kg = float(row[3]) if row[3] is not None else 0
            
            materials_info.append({
                "material": material,
                "description": description,
                "hazard_division": hd,
                "new_per_unit_kg": new_kg
            })
            
            if hd:
                hd_weights[hd] = hd_weights.get(hd, 0) + new_kg
                if hd not in materials_by_hd:
                    materials_by_hd[hd] = []
                materials_by_hd[hd].append({
                    "material": material,
                    "description": description,
                    "new_kg": new_kg
                })
        
        if not hd_weights:
            return {"error": "No valid hazard division data found for specified materials"}
        
        # Apply IATG aggregation
        aggregator = IATGAggregator(hd_weights)
        aggregated_hd, aggregated_new = aggregator.calculate_aggregation(verbose=False)
        
        # Prepare HD breakdown
        hd_breakdown = []
        for hd, weight in hd_weights.items():
            hd_breakdown.append({
                "hazard_division": hd,
                "total_new_kg": round(weight, 4),
                "material_count": len(materials_by_hd.get(hd, [])),
                "materials": materials_by_hd.get(hd, [])
            })
        
        # Check for missing materials
        found_materials = {row[0] for row in results}
        missing_materials = [m for m in material_numbers if m not in found_materials]
        
        return {
            "materials_analyzed": materials_info,
            "material_count": len(materials_info),
            "hazard_division_breakdown": hd_breakdown,
            "iatg_aggregated_hd": aggregated_hd,
            "iatg_aggregated_new_kg": round(aggregated_new, 4),
            "iatg_rule_applied": f"If stored together, these materials would be aggregated to HD {aggregated_hd} per IATG guidelines",
            "missing_materials": missing_materials if missing_materials else None,
            "usage": "Use aggregated values for Quantity Distance (QTD) calculations if these materials are stored together"
        }
        
    except Exception as e:
        return {"error": f"Error calculating materials aggregation: {str(e)}"}


# ============================================================================
# TOOL DEFINITIONS FOR LLM
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_compatibility_info",
            "description": "Get comprehensive compatibility information for a specific explosive material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_number": {
                        "type": "integer",
                        "description": "The material number (e.g., 885600034)"
                    }
                },
                "required": ["material_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database_with_sql",
            "description": "Execute a natural language query against the explosives master data in HANA. Note: This only accesses master data tables (HMSD, CG, HD, SLOC). For current inventory quantities and storage locations, use get_material_inventory or get_storage_summary instead. Use this for queries about material properties, compatibility groups, hazard divisions, or storage location descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "Natural language description of what master data to retrieve from the database"
                    }
                },
                "required": ["user_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_inventory",
            "description": "Get detailed LIVE inventory information for a specific material across all storage locations from S/4 HANA, including current quantities, batches, and total NET Explosive Weight (NEW). Use this when user asks about inventory levels, stock, or where a specific material is currently stored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_number": {
                        "type": "integer",
                        "description": "The material number to check inventory for"
                    }
                },
                "required": ["material_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_materials",
            "description": "List all explosive materials in the master data with their basic properties (material number, description, hazard division, compatibility group, NEW per unit). Note: This shows what materials exist but NOT current inventory quantities. Use get_material_inventory to check actual stock levels.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_storage_summary",
            "description": "Get a summary of all storage locations with current total quantities, NEW values, and IATG-aggregated NEW for QTD calculations from LIVE S/4 HANA data. Use this when user asks about storage capacity, utilization, or wants an overview of where explosives are currently stored. Includes aggregated Hazard Division per IATG rules.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_storage_location_qtd",
            "description": "Calculate the IATG-aggregated Hazard Division and NEW for a specific storage location. This applies International Ammunition Technical Guidelines (IATG) rules to determine the aggregated HD and NEW used for Quantity Distance (QTD) and safety distance calculations. Use this when user asks about QTD, safety distances, or what HD to use for a specific storage location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_location": {
                        "type": "string",
                        "description": "The storage location code (e.g., '00AJ', '00AK')"
                    }
                },
                "required": ["storage_location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_materials_aggregation",
            "description": "Calculate the IATG-aggregated Hazard Division and NEW for a specific list of materials. This 'what-if' analysis tool determines what the aggregated HD and NEW would be if the specified materials were stored together. Use this when user asks about combining specific materials, planning storage, or wants to know the aggregated NEW/HD for multiple materials. Example: 'What if I store materials 885600034 and 885600011 together?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_numbers": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "List of material numbers to aggregate (e.g., [885600034, 885600011])"
                    }
                },
                "required": ["material_numbers"]
            }
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "get_compatibility_info": get_compatibility_info_tool,
    "query_database_with_sql": query_database_with_sql_tool,
    "get_material_inventory": get_material_inventory_tool,
    "list_all_materials": list_all_materials_tool,
    "get_storage_summary": get_storage_summary_tool,
    "calculate_storage_location_qtd": calculate_storage_location_qtd_tool,
    "calculate_materials_aggregation": calculate_materials_aggregation_tool
}


# ============================================================================
# MAIN AGENT FUNCTION
# ============================================================================

def run_explosives_agent(
    user_question: str,
    max_iterations: int = 10,
    model: str = "gpt-5",
    verbose: bool = True
) -> str:
    """
    Run the explosives intelligence agent to answer user questions.
    
    Args:
        user_question: The user's question about explosives
        max_iterations: Maximum number of tool calls allowed
        model: Model to use (gpt-5, gpt-4o, gpt-35-turbo, etc.)
        verbose: Print tool calls and intermediate steps
    
    Returns:
        The agent's final answer as a string
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
    
    if verbose:
        print("EXPLOSIVES INTELLIGENCE AGENT")
        print("=" * 80)
        print(f"Question: {user_question}\n")
    
    for iteration in range(max_iterations):
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
            final_answer = response_message.get("content")
            if verbose:
                print(f"\nFinal Answer:\n{final_answer}")
                print("\n" + "=" * 80)
            return final_answer
        
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            if verbose:
                print(f"Tool Call: {function_name}")
                print(f"   Parameters: {function_args}")
            
            try:
                function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
                if verbose:
                    print(f"   ✓ Response received ({len(json.dumps(function_response))} chars)")
            except Exception as e:
                function_response = {"error": str(e)}
                if verbose:
                    print(f"   ✗ Error: {str(e)}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": function_name,
                "content": json.dumps(function_response)
            })
            
            if verbose:
                print()
    
    return "Maximum iterations reached. Could not complete the query."
