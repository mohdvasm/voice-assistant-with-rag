import os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import sys
import warnings

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class Assistant:
    def __init__(self, use_groq: bool = True, use_ollama: bool = False):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Help the user with your helpful response. Your response could be related to context or just usual conversation regardless of the context. If user query relates to the given context, then only use context. \nContext: {context}",
                ),
                ("user", "{query}"),
            ]
        )

        # if use_groq:
        #     self.model = self.get_groq_model()
        # else:
        #     self.model = self.get_ollama_model()

        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        )
        
        self.retriever = None 
        self.documents = None 
        self.chain = None 

    @staticmethod
    def get_pdf_content(file_path: str) -> list:
        """Load a PDF file and return its content as a list."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def get_text_content(file_path: str) -> list:
        """Load a text file and return its content as a list."""
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    def get_retriever_from_files(
            self, 
            files: list,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            search_type: str = "mmr",
            search_kwargs: dict = {"k": 1},
        ) -> FAISS:
            """Load multiple PDF files and create a retriever using FAISS."""

            try:
                all_documents = []
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                documents = []

                for file_path in files:
                    if file_path.endswith('.txt'):
                        documents = self.get_text_content(file_path)
                    elif file_path.endswith('.pdf'):
                        documents = self.get_pdf_content(file_path)
                    else:
                        raise ValueError(f"Unsupported file type: {file_path}")
                    
                    all_documents.extend(documents)
                
                print(f"Loaded {len(all_documents)} documents from {len(files)} files.")

                all_documents = text_splitter.split_documents(all_documents)

                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                vector_store = FAISS.from_documents(all_documents, embeddings)
                
                retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
                
                self.retriever = retriever
                self.documents = all_documents

                self.build_chain()
                return retriever
            
            except Exception as e:
                print(f"Error: {e}")
                self.build_chain()

    def build_chain(self):
        self.chain = self.prompt | self.model
        return self.chain

    def get_ollama_model(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0,
        max_tokens: int = 100,
    ):
        return ChatOllama(
            model=model_name,
            temperature=temperature
            # other params...
        )
    
    def get_groq_model(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.5,
        max_tokens: int = 256,
        timeout: int = 60,
        max_retries: int = 2,
    ):
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            # other params...
    )
    
    def __call__(self, query: str):
        """Call the chain with a query and return the result."""
        if self.chain is None:
            raise ValueError("Chain has not been built yet.")
        
        context = self.retriever.invoke(query)
        result = self.chain.invoke({"query": query, "context": context})
        result, content = result, result.content
        return result, content
