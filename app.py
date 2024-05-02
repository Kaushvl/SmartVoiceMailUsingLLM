import streamlit as st
from initialVersion import HandleCallInput
import os
import pymongo
import pandas as pd

mongo_uri = os.environ.get('MONGO_URI')  # Replace with your MongoDB connection string

def connect_to_mongo():
  """Connects to the MongoDB database"""
  client = pymongo.MongoClient(mongo_uri)
  db = client['CallSum']
  Collection = db['CallData']
  return Collection

def fetch_all_data():
    """Fetches all call data from the database"""
    collection = connect_to_mongo()
    data = list(collection.find())
    return data

def main():
    st.title("Call Processing App")

    # Audio file upload option
    uploaded_file = st.file_uploader("Upload audio file")
    if uploaded_file is not None:
        # Save uploaded file (optional)
        strfilepath = os.path.join("uploads", uploaded_file.name)
        with open(strfilepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcribe audio
        str_audio_content = HandleCallInput(strfilepath)  # Assuming temporary storage

    # Process transcribed text
        all_data = fetch_all_data()

        # Check if data is available
        if all_data:
            # Extract relevant fields and create headers
            headers = ["Call Urgency", "Call Title", "Call Summary","DateTime"]
            call_data = []
            for entry in all_data:
                call_data.append([entry["CallUrgency"], entry["CallTitle"], entry["CallSummary"],entry["DateTime"]])

            dfData = pd.DataFrame(call_data,columns=headers)

            # Display data in a table
            st.table(dfData.sort_values(by='DateTime',ascending=False).reset_index(drop=True))
        else:
            st.info("No call data found in the database.")



if __name__ == "__main__":
    main()