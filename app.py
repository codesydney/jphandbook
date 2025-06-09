# app.py
from flask import Flask, request, jsonify, render_template
import lancedb
import time
import os
import sys
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from step_1_embedding import prepare_data  

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set models
llm = GoogleGenAI(model="gemini-2.0-flash", temperature=0.2)  # Slightly increased temperature for better answers
embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

Settings.llm = llm
Settings.embed_model = embed_model

app = Flask(__name__)

# Configuration
db_path = 'lancedb'
table_name = 'jp_handbook'

# Function to query the database directly
def query_database(question, top_k=8):  # Increased from 5 to 8 to get more context
    try:
        # Connect to the database
        db = lancedb.connect(db_path)
        
        # Check if table exists
        if table_name not in db.table_names():
            print(f"Table '{table_name}' not found. Running prepare_data() to create it...")
            prepare_data()
            
            # Check again after preparation
            db = lancedb.connect(db_path)
            if table_name not in db.table_names():
                return {"error": f"Table '{table_name}' could not be created"}, None
        
        # Open the table
        table = db.open_table(table_name)
        
        # Create embedding for the question
        question_embedding = embed_model.get_text_embedding(question)
        
        # Query the table with more results
        results = table.search(question_embedding).limit(top_k).to_pandas()
        
        # Process results - deduplicate similar content
        context_texts = []
        seen_content = set()
        
        for text in results['text'].tolist():
            # Simple deduplication check - skip if very similar to something we've seen
            content_hash = hash(text[:100])  # Use first 100 chars as a signature
            if content_hash not in seen_content:
                context_texts.append(text)
                seen_content.add(content_hash)
        
        # Join context with clear section markers
        context = "\n\n--- Document Section ---\n".join(context_texts)
        
        # Improved prompt with structured reasoning
        prompt = f"""
        You are a Japan travel expert assistant helping someone with their Japan travel questions.
        Your goal is to provide accurate, helpful, and concise answers based only on the provided document sections.
        
        CONTEXT INFORMATION (from JP Handbook):
        {context}
        
        QUESTION: {question}
        
        INSTRUCTIONS:
        1. Analyze the context information thoroughly
        2. Only use information from the provided context
        3. If the answer isn't in the context, say "I don't have specific information about that in the handbook"
        4. Provide specific details when available (names, places, prices, procedures)
        5. Be conversational but concise
        
        ANSWER:
        """
        
        # Generate answer with improved prompt
        response = llm.complete(prompt)
        
        # Process sources for display
        sources = []
        for i, row in results.iterrows():
            # Extract more context for better understanding
            text = row['text']
            preview = text[:300] + "..." if len(text) > 300 else text
            
            sources.append({
                "text": preview,
                "score": f"{row['_distance']:.4f}" if '_distance' in row else "N/A"
            })
        
        return str(response), sources
    
    except Exception as e:
        print(f"Error in query_database: {e}")
        return f"Error: {str(e)}", None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    question = request.form.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        start_time = time.time()
        answer, sources = query_database(question)
        processing_time = time.time() - start_time
        
        if isinstance(answer, dict) and "error" in answer:
            return render_template('error.html', error=answer["error"])
        
        return render_template('results.html',
                            question=question,
                            answer=answer,
                            sources=sources if sources else [],
                            processing_time=f"{processing_time:.2f}")
    
    except Exception as e:
        print(f"Error in handle_query: {e}", file=sys.stderr)
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)