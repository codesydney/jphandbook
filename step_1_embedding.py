import os
from dotenv import load_dotenv
import lancedb
import numpy as np
import pandas as pd
import time

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Load environment variables
load_dotenv()

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration
pdf_path = 'jp-handbook-full.pdf'
db_path = 'lancedb'
table_name = 'jp_handbook'

# Initialize models
llm = GoogleGenAI(model="gemini-2.0-flash", temperature=0.1)
embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

# Set global settings for llama_index
Settings.llm = llm
Settings.embed_model = embed_model

def prepare_data():
    print("Preparing data...")
    
    # Ensure db_path directory exists
    os.makedirs(db_path, exist_ok=True)
    
    # Step a: Load documents
    print(f"Loading documents from {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return
    
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    print(f"✅ Loaded {len(documents)} documents")

    # Step b: Split into nodes
    print("Splitting documents into nodes...")
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"✅ {len(nodes)} nodes created.")

    # Step c: Connect to LanceDB directly
    print(f"Connecting to LanceDB at {db_path}...")
    try:
        db = lancedb.connect(db_path)
        print(f"✅ Connected to LanceDB")
        print(f"Existing tables in LanceDB: {db.table_names()}")
    except Exception as e:
        print(f"❌ Error connecting to LanceDB: {e}")
        return

    # Step d: Drop table if it already exists
    if table_name in db.table_names():
        print(f"Dropping existing table '{table_name}'...")
        db.drop_table(table_name)
        print(f"✅ Table '{table_name}' dropped")

    # Step e: Directly create embeddings and table
    print("Creating embeddings directly...")
    data = []
    
    for i, node in enumerate(nodes):
        try:
            # Print progress for every 10 nodes
            if i % 10 == 0:
                print(f"Embedding node {i}/{len(nodes)}...")
            
            text = node.get_content()
            node_id = node.node_id
            
            # Get embedding from the embed model
            embedding = embed_model.get_text_embedding(text)
            
            # Create a data entry
            entry = {
                "id": node_id,
                "text": text,
                "vector": embedding,
                "metadata": {
                    "doc_id": node.ref_doc_id if hasattr(node, 'ref_doc_id') else "",
                }
            }
            data.append(entry)
            
        except Exception as e:
            print(f"Error embedding node {i}: {e}")
    
    print(f"✅ Created {len(data)} embeddings")
    
    # Step f: Create DataFrame and write directly to LanceDB
    try:
        print(f"Creating table '{table_name}' directly...")
        
        # Check if we have valid data
        if not data:
            print("❌ No valid data to create table with")
            return
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Create LanceDB table
        db.create_table(
            table_name,
            data=df,
            mode="overwrite"
        )
        
        print(f"✅ Table '{table_name}' created directly in LanceDB")
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return

    # Final check
    try:
        updated_tables = db.table_names()
        print(f"Updated tables: {updated_tables}")
        if table_name in updated_tables:
            print(f"✅ Table '{table_name}' was successfully created.")
            # Get table info for debugging
            table = db.open_table(table_name)
            print(f"Table schema: {table.schema}")
            print(f"Table row count: {table.count_rows()}")
        else:
            print(f"❌ Table '{table_name}' was NOT created.")
    except Exception as e:
        print(f"❌ Error checking table: {e}")

    print("Data preparation complete.")

if __name__ == "__main__":
    start_time = time.time()
    prepare_data()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")