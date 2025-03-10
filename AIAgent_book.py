import streamlit as st
import faiss
import pandas
import re

from ragatouille import RAGPretrainedModel
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores import FAISS, Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


import os
import pandas as pd

import csv


# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
import os
# Load the API key from Streamlit secrets
api_key = st.secrets["API_KEY"]
genai.configure(api_key=api_key)

import streamlit as st
import pandas as pd
import re
import google.generativeai as genai

from ragatouille import RAGPretrainedModel
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores import FAISS, Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Load the API key from Streamlit secrets
api_key = st.secrets["API_KEY"]
genai.configure(api_key=api_key)

class AIAgent_book:
    def __init__(self, max_length=256, num_retrieved_docs=3):
        self.max_length = max_length
        self.num_docs = num_retrieved_docs
        self.model = genai.GenerativeModel('gemini-pro')

        df1 = pd.read_csv("book_dataset.csv", index_col=0)
        df1.reset_index(drop=True, inplace=True)

        # Clean the Book Title and handle missing values in Book Content
        df1['Book Title'] = df1['PDF Name'].apply(lambda x: re.sub(r'\.pdf$', '', x))
        df1 = df1.drop(columns=['PDF Name'])

        # Fill missing values in 'Book Content' with an empty string or any placeholder
        df1['Book Content'].fillna('', inplace=True)

        loader1 = DataFrameLoader(df1, page_content_column="Book Content")
        documents1 = loader1.load()

        text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = text_splitter1.split_documents(documents1)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        num_docs = 5  # Default number of documents to retrieve

        bm25_retriever = BM25Retriever.from_documents(chunked_docs).configurable_fields(
            k=ConfigurableField(id="search_kwargs_bm25", name="k", description="The search kwargs to use")
        )

        faiss_vectorstore = FAISS.from_documents(chunked_docs, embedding_model, distance_strategy=DistanceStrategy.COSINE)

        faiss_retriever = faiss_vectorstore.as_retriever(
            search_kwargs={"k": num_docs}
        ).configurable_fields(
            search_kwargs=ConfigurableField(id="search_kwargs_faiss", name="Search Kwargs", description="The search kwargs to use")
        )

        # Initialize the ensemble retriever
        self.vector_database = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )

        self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

        prompt_template = """  
        You are an AI agent specializing in providing answers based on the context given from the books. If the exact answer is not available in the context, you should analyze the information and provide an approximate answer.
        Question: {query}
        Context: {context}

        Answer:
        """

        self.RAG_PROMPT_TEMPLATE = PromptTemplate(
            input_variables=["query", "context"],
            template=prompt_template,
        )

    def answer_with_rag(
        self,
        question: str,
        num_retrieved_docs: int = 10,
        num_docs_final: int = 5,
    ):
        # Gather documents with retriever
        print("=> Retrieving documents...")
        config = {"configurable": {"search_kwargs_faiss": {"k": num_retrieved_docs}, "search_kwargs_bm25": num_retrieved_docs}}
        relevant_docs = self.vector_database.invoke(question, config=config)
        relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

        print("=> Reranking documents...")
        relevant_docs = self.reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]  # Keeping only num_docs_final documents

        # Build the final prompt
        context = relevant_docs[0]  # We select only the top relevant document

        final_prompt = self.RAG_PROMPT_TEMPLATE.format( 
            query=question,
            context=context
        )

        # Generate an answer
        print("=> Generating answer...")
        answer = self.model.generate_content(final_prompt)

        return answer, relevant_docs

