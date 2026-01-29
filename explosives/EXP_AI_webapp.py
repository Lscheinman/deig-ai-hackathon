import gradio as gr
import json
import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

api_url = "https://explosives-api.cfapps.ap10.hana.ondemand.com/api/agent/query"
headers = {'Content-Type': 'application/json'}

def get_selection(choice):
    return f"You selected: {choice}"

def fetch_data_inventory(api_url):
    headers = {'authorization': 'Basic QVBJVVNSOldlbGNvbWUx' , 'accept' : 'application/json'}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes
    data = response.json()
    results_data = data['d']
    return results_data

def fetch_data_HMSD(api_url):
    headers = {'accept' : 'application/json'}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes
    data = response.json()
    return data


def merge_api_data(merged_df,slocs):
    # 1. Fetch Data
    inventory_url = "https://coeportal515.saphosting.de/sap/opu/odata/sap/API_MATERIAL_STOCK_SRV/A_MatlStkInAcctMod?sap-client=600&$filter="
    
    number_of_slocs = len(slocs)
    sloc_filter = "((MatlWrhsStkQtyInMatlBaseUnit gt 0) and ("
    for i in range(number_of_slocs):
        if i == 0 :
            sloc_filter = sloc_filter + "StorageLocation eq '" + slocs[i] + "'" 
        else:
            sloc_filter = sloc_filter + "or StorageLocation eq '" + slocs[i] + "'"     

    sloc_filter = sloc_filter + "))"

    
    inventory_url = inventory_url + sloc_filter
    
    inventory_data = fetch_data_inventory(inventory_url) # Replace with your user API
    HSMD_data = fetch_data_HMSD("https://explosives-api.cfapps.ap10.hana.ondemand.com/api/materials") 

    # 2. Convert to Pandas DataFrames
    inv_columns_to_show = ['Material','StorageLocation','Batch','MatlWrhsStkQtyInMatlBaseUnit']
    df_inventory = pd.DataFrame(inventory_data['results'])
    df_inventory = df_inventory[inv_columns_to_show]
    df_inventory = df_inventory.astype(str)
    
    df_HSMD = pd.DataFrame(HSMD_data['materials'])
    df_HSMD = df_HSMD.astype(str)
    

    # 3. Merge DataFrames (assuming 'user_id' is the common key)
    # Use how='inner', 'left', 'right', or 'outer' as needed
    merged_df = pd.merge(df_inventory,df_HSMD, left_on='Material', right_on='material_number', how='left')
    merged_df["MatlWrhsStkQtyInMatlBaseUnit"] = pd.to_numeric(merged_df["MatlWrhsStkQtyInMatlBaseUnit"])
    merged_df["new_kg"] = pd.to_numeric(merged_df["new_kg"])
    merged_df["total"] = merged_df["MatlWrhsStkQtyInMatlBaseUnit"] * merged_df["new_kg"]
    merged_df["MatlWrhsStkQtyInMatlBaseUnit"] = pd.to_numeric(merged_df["MatlWrhsStkQtyInMatlBaseUnit"])
    columns_to_show = ['StorageLocation','Material','description','Batch','hazard_division','compatibility_group','new_kg','MatlWrhsStkQtyInMatlBaseUnit','total']
    merged_df = merged_df[columns_to_show]
    merged_df.columns = ['SLOC','Material','Description','Batch','HD','CG','NEW (KG) for EA','QTY (EA)','NEW (KG) for QTY']

    # 4. Return the merged DataFrame (Gradio handles conversion for gr.DataFrame)
    return merged_df


with gr.Blocks() as demo:
    slocs = gr.Dropdown(
        choices=["00AJ", "00AK", "00AL", "00AM", "00AN", "00AO", "00AP", "00AQ", "00AR"],
        value =["00AJ"],
        label="What SLOCs are you interested in?",
        info="Choose one or more of the following",
        multiselect= True
    )
    
    # output = gr.Textbox(label="Output")
    
    # Triggered when user selects a new value
    slocs.input(fn=get_selection, inputs=slocs, outputs=None)
    output_table = gr.DataFrame(headers=['SLOC','Material','Description','Batch','HD','CG','NEW (KG) for EA','QTY (EA)','NEW (KG) for QTY'], interactive=False)
    btn = gr.Button("Load Inventory and HMSD Data filtered to your selected SLOCs")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask me something related to Explosives Management in your IM storage locations...", submit_btn="Send")
    with gr.Row():
        clear_btn = gr.ClearButton([msg, chatbot])
        
    with gr.Row():
        CG_btn = gr.Button("Generate a CG Compatibility Compliance report for each the selected SLOCs",variant="primary")
        NEW_btn = gr.Button("Calculate aggregated NEW for each selected SLOCs",variant="primary")

    def user(user_message, history: list):
        return user_message, history + [{"role": "user", "content": user_message}]
    
    def bot(user_message, history: list):
        data_to_send = "{\"question\": \""+user_message+"\",\"model\": \"gpt-5\",\"max_iterations\": 20}"
        payload = json.loads(data_to_send) 
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), verify=False)
        data_answer_all = response.json()
        bot_message = data_answer_all.get("answer","No answer found")

        return user_message, history + [{"role": "assistant", "content": bot_message}]





    def update_CG_text(choices):
    # Return a new value to update the component
        return f"Generate a Compatibility Compliance report for each of the following SLOC: {", ".join(choices)}"

    
    def update_NEW_text(choices):
    # Return a new value to update the component
        return f"Provide an aggregated NEW report on SLOC: {", ".join(choices)}"
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(bot, [msg,chatbot], [msg,chatbot])
    
    CG_btn.click(fn=update_CG_text, inputs=slocs, outputs=msg)
    NEW_btn.click(fn=update_NEW_text, inputs=slocs, outputs=msg)
    btn.click(fn=merge_api_data, inputs=[output_table,slocs], outputs=output_table)
    
demo.launch(server_name="0.0.0.0")
# demo.launch(share=True)
