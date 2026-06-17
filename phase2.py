import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "google/gemma-4-31B-it"

def load_llm(huggingface_repo_id):
    from huggingface_hub import InferenceClient
    from langchain_core.language_models.llms import LLM
    from typing import Optional, Any
    from pydantic import Field

    class CustomHFLLM(LLM):
        model: str = Field(default="")
        token: Optional[str] = Field(default=None)
        _client: Any = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            object.__setattr__(
                self, '_client',
                InferenceClient(model=self.model, token=self.token)
            )

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=1
            )
            return response.choices[0].message.content

        @property
        def _llm_type(self):
            return "custom_hf"

        @property
        def _identifying_params(self):
            return {"model": self.model}

    return CustomHFLLM(model=huggingface_repo_id, token=HF_TOKEN)


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

import sys
if not sys.stdin.isatty():
    user_query = sys.stdin.read().strip()
else:
    user_query = input("Write Query Here: ")

response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])