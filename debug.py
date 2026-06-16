import warnings
warnings.filterwarnings("ignore")

print("Step 1: imports...")
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

print("Step 2: token check...")
HF_TOKEN = os.environ.get("HF_TOKEN")
print("Token:", HF_TOKEN[:10] if HF_TOKEN else "NONE")

print("Step 3: loading embeddings...")
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings loaded!")

print("Step 4: loading FAISS...")
from langchain_community.vectorstores import FAISS
db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
print("FAISS loaded!")

print("Step 5: loading LLM...")
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.5, model_kwargs={"token": HF_TOKEN, "max_length": 512})
print("LLM loaded!")

print("ALL DONE!")