# jphandbook
NSW Justice of the Peace Handbook


# RAG System using LlamaIndex and LanceDB
This notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) system for question answering on the jp-handbook-full.pdf document using:

- LlamaIndex for document processing and querying
- LanceDB as the vector store

# Summary 
1. Install all needed Python libraries like LlamaIndex, LanceDB, pypdf, dotenv, and Google GenAI.
2. Import the libraries and load your Google API key from the .env file.
3. Set up the LLM (Gemini) and embedding model (text-embedding-004) using your API key.
4. Check if the jp-handbook-full.pdf file exists and load it into memory.
5. Split the loaded document into smaller text chunks (nodes) using a sentence splitter.
6. Connect to LanceDB, drop old tables if needed, and create a new vector database table.
7. Build a searchable vector index from the text chunks (nodes).
8. Done! Your PDF is now loaded, chunked, and stored in LanceDB for fast search.

# Steps 

### 1. Install Required Dependencies
##### Notes:
    - Inside VS Code, I have to install the Jupyter Notebook extension, it installs a kernel and I have to choose Python environment instead of Jupyiter server.
    - I also signed up for a Google Gemini API as we will use Gemini here.
    - Installation of required libraries. 
    - It took few seconds to install these dependencies.

### 2. Import Libraries
##### Notes:
    - Import of required libraries. 
    - It took few seconds to install these dependencies.

### 3. Load and Process the PDF Document
##### Notes:
    - The PDF is 2.7 MB in size with 132 pages.
    - After the run, it says "Loaded 132 document(s)". Looks like the default chunking is per page. 
    - It took around 7s to run this step. 
    - Question? Where did it load the chunks? In-memory perhaps?

### 4. Parse the Document into Nodes
##### Notes:
    - After the run, it says "Document split into 132 nodes". It appears default node is a page in this case. 
    - It took 0.4s to finish. 

### 5. Set Up LanceDB Vector Store
##### Notes:
    - It says Vector index created successfully but can't seem to find any persisted file.
    - It took around 10s to finish. 
    
### 6. Create a Query Engine
##### Notes:
    - Just prepping LlamaIndex querying.
    - It took subsecond to finish.
    
### 7. Ask Questions
##### Notes:
    - Actual Q&A.
    - It took subsecond to finish.

### 8. Multiple Q&A
##### Notes:
    - Actual Q&A.
    - It took subsecond to finish.
    
### 9. Interactive Q&A
##### Notes:

### 10. Conclusion
##### Notes:

## License

[MIT](https://choosealicense.com/licenses/mit/)
