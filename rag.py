import nltk
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import pandas as pd

class RAG:
    def __init__(self):
        self._download_nltk()

        self.key = 'AIzaSyBkeAaLuUE8mkyDSNdNc6ULcVTqjfcx-ro'
        self.chat_model = ChatGoogleGenerativeAI(google_api_key=self.key, model="gemini-2.0-flash")
        self.embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=self.key, model="models/embedding-001")

        self.pages = None
        self.chunks = None
        self.retriever = None
        self.rag_chain = None

    def _download_nltk(self):
        for pkg in ['punkt', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                nltk.download(pkg)

    def load_from_pdf(self, path):
        try:
            loader = PyPDFLoader(path)
            self.pages = loader.load_and_split()
            self._process_documents()
        except Exception as e:
            print(f"[ERROR] Failed to read PDF: {e}")

    def load_from_website(self, url):
        try:
            loader = WebBaseLoader(url)
            self.pages = loader.load()
            self._process_documents()
        except Exception as e:
            print(f"[ERROR] Failed to read website: {e}")

    def load_from_csv(self, path):
        """Load and process data from a CSV file with structured content."""
        try:
            df = pd.read_csv(path)
            
            # Create documents where each document represents a row with explicit columns
            self.pages = []
            for _, row in df.iterrows():
                page_content = ""
                for col_name, value in row.items():
                    page_content += f"{col_name}: {value}\n"
                self.pages.append(Document(page_content=page_content.strip()))
            
            if not self.pages:
                raise ValueError("No data found in CSV.")
            
            self._process_documents()
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}")

    def _process_documents(self):
        self.text_splitter = NLTKTextSplitter(chunk_size=4000, chunk_overlap=1000)
        self.chunks = self.text_splitter.split_documents(self.pages)

        self.db = Chroma.from_documents(self.chunks, self.embedding_model, persist_directory="./chroma_db_")

        self.retriever = self.db.as_retriever(search_kwargs={"k": 500})
        self.setup_chat_template()

    def setup_chat_template(self):
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful assistant. Use the context to answer the user's question.
            Give the answer without introduction. Provide an answer based on your information if context does not provide answers."""),
            HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
        ])

        output_parser = StrOutputParser()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | chat_template
            | self.chat_model
            | output_parser
        )

    def chat(self, query):
        if not self.rag_chain:
            return "[ERROR] No data loaded. Please load PDF, website, or CSV content first."
        try:
            return self.rag_chain.invoke(query)
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")
            return "Sorry, an error occurred."
        
# LLM = RAG()
# LLM.load_from_csv("jordan_transactions.csv")
# answer = LLM.chat("Hi")
# print(answer)
        