# Invest Chat  - simple RAG demo app

** This chat is designed to answer questions exclusively using data from the datasource. If the required data is missing from the datasource, the LLM will respond with "I don't know.". If you need clear chat history, ask assistant to do it. **

Invest Chat is a project written in Python 3.8. Before starting, copy `.env_example` to a new file named `.env` and add your `OPENAI_API_KEY`.  

---

## How to Run  

### Create a Virtual Environment  

#### Using Conda  
To install Conda, refer to the official guide: [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).  

Replace `ENV_NAME` with your desired environment name (e.g., `rag`). Then, initialize and activate the Conda environment:  

```bash
conda init
conda create -n ENV_NAME python=3.8
conda activate ENV_NAME
```  

To deactivate the environment, run:  
```bash
conda deactivate
```  

---

### Install Packages  
Install the required dependencies with the following command:  
```bash
pip install -r requirements.txt
```  

---

### Run the Application  

#### First Run and Reindexing  
When new PDF files are added to the `/datasources` folder, use this command to extract and index the data:  
```bash
python main.py --run-extractor
```  

#### Running the Chat Application  
If the datasources have already been vectorized, you can simply run the chat application:  
```bash
python main.py
```  

---

## Folder Structure  

### `/datasources`  

The `/datasources` folder is used to store PDF documents. This repository does not include a pre-built PDF knowledge base. Before using the application, add investment-related PDFs to the `/datasources` folder.  

You can find relevant PDFs by searching online, for example:  
[How to Invest PDF Search](https://www.google.com/search?q=How+to+invest+filetype%3Apdf)  

After adding your PDFs, process them by running:  
```bash
python main.py --run-extractor
```  

---

### `/database`  

The `/database` folder serves as storage for ChromaDB. Vectorized data is stored in this folder.  

To reset the vectorized data, simply delete the contents of the `/database` folder. The database will be recreated when you run:  
```bash
python main.py --run-extractor
```  

---

### `/src`  

The `/src` folder contains all custom Python modules. Below is an overview of its contents:  

- `/src/webserver.py`: Flask web server for the application.  
- `/src/templates/`: Directory for Flask `.html` templates.  
- `/src/pdf_extractor.py`: Script to extract data from PDFs.  
- `/src/embeddings.py`: Script to convert extracted data into vectors and store them in ChromaDB.  
- `/src/retrieve.py`: Script to search for vectors similar to the user’s query.  
- `/src/ai_service.py`: Script to create the system prompt, attach the retrieved context, and send a payload to OpenAI.
- `/src/chat_history.py`: User chat history object.
