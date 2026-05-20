import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma #uv add langchain_chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter # uv add langchain_text_splitters
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # uv add langchain_classic

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from .structlogger import logger 

# =======================================+++==================
# Data Model - forces reply with a  JSON structure
# ============================================================
class ResearchResponse(BaseModel):
    """Structured response from the research assistant."""
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low based on source quality")
    sources: List[str] = Field(description="List of source documents used")
    key_quotes: List[str] = Field(
        description="Relevant quotes from sources", default=[]
    )
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

# ============================================================
# Research Assistant Class
# ============================================================

class AIResearchAssistant:
    """AI Research Assistant."""

    def __init__(
        self,
        persist_directory :str, #: str = "./research_db",
        chunk_size: int,        # = 1000,
        chunk_overlap: int,     # = 200,
    ):  
        self.persist_directory = persist_directory

        # 1. Embeddings - turns text into vectors
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Handles converting human language chunks into high-dimensional vector arrays

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Foundational language model

        # 2. Splitter - breaks big docs into chunks
        # Tries to split text by paragraphs (\n\n) first. If a paragraph is still too big, it falls back to lines (\n), then sentences (. ), then words ( )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # 3. Vector store - stores and searches embeddings
        # Establishes a connection to local, light-weight vector database
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings, # Instructs DB to automatically use OpenAI embeddings model to convert queries into vectors
            collection_name="research_docs",
        )

        # Dictionary to tracki conversation history
        self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

        print(f"AIResearchAssistant  class instantiated")
        print(f"Vector store persist_directory: {os.path.abspath(self.persist_directory)}")
        print(f"chunk_size: {chunk_size}")
        print(f"chunk_overlap: {chunk_overlap}")
        print(f"Number of indexed documets: {self.vectorstore._collection.count()}")
        # Logger
        logger.info(f"AIResearchAssistant  class instantiated")
        logger.info("Vector store persist_directory: ", persist_directory=os.path.abspath(self.persist_directory))
        logger.info("chunk_size: ", chunk_size=chunk_size)
        logger.info("chunk_overlap: ", chunk_overlap=chunk_overlap)
        logger.info("Number of indexed documets: ", vectorstore_count=self.vectorstore._collection.count())

    def add_documents(
        self,
        documents: List[Document],
        source_name: Optional[str] = None,
    ) -> int:
        """Add documents to the research database."""

        # Tag with source name
        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name
        
        # Split into chunks
        chunks = self.splitter.split_documents(documents)

        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        # Store chunks in vector DB
        # Takes each chunk's text, sends it to the embedding model's API, receives a high-dimensional vector array representing its meaning
        # Finally saves both the vector and the raw text to the persistance storage
        self.vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks) 

    def add_text(self, text: str, source: str, metadata: dict = None) -> int:
        """Add a single text string as a document."""
        doc = Document(
            page_content=text, metadata={"source": source, **(metadata or {})}
        )
        return self.add_documents([doc])
    
    def add_texts(self, texts: List[str], source: str) -> int:
        """Add multiple text strings from the same source."""
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        return self.add_documents(docs)

    def get_document_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.vectorstore._collection.count()
    
    # Return a sorted list of all unique document sources stored
    def list_sources(self) -> List[str]:
        """List all unique sources in the database."""
        results = self.vectorstore._collection.get()
        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        return sorted(list(sources))
    
    # Information extraction engine for RAG pipeline - private helper method
    def _build_retriever(self, use_advanced: bool = False):
        """Build retriever -- basic or advanced"""

        # Base: simple similarity search
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}    
        )

        if not use_advanced:
            return base_retriever

        # Multi-query: LLM generates multiple search queries based on source query
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,    
        )    

        return multi_retriever
    
    def _format_docs_for_context(self, docs) -> str:
        """Format retrieved documents into a string for the prompt."""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    async def ask_structured(
        self,
        question: str,
        session_id: str = "default",
        use_advanced: bool = True,                
    ) -> ResearchResponse:
        """Ask a question and get a structured response."""

        # LLM that returns a Pydantic object instead of a string
        structured_llm = self.llm.with_structured_output(ResearchResponse)

        # Get history
        history = self._get_session_history(session_id)

        # Retrieve
        retriever = self._build_retriever(use_advanced=use_advanced)
        #docs = retriever.invoke(question)
        docs = await retriever.ainvoke(question)
        context = self._format_docs_for_context(docs)
        sources = list(set(d.metadata.get("source", "Unknown") for d in docs))

        # Prompt -- explain the LLM available sources
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Analyze the provided documents 
    and return a structured response.

    Rules:
    1. ONLY use information from the provided context
    2. If the context doesn't have the answer, say so in the answer field
    3. Set confidence: "high" if directly stated, "medium" if inferred, "low" if partial
    4. Include the source filenames you actually used
    5. Extract key quotes word-for-word from the context
    6. Suggest 2-3 follow-up questions the user might want to ask

    Use conversation history to understand follow-up questions.""",                    
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Available sources: {sources}

    Question: {question}""",
                ),
            ]
        )

        chain = prompt | structured_llm

        #response = chain.invoke(
        response = await chain.ainvoke(
            {
                "context": context,
                "question": question,
                "sources": ", ".join(sources),
                "history": (
                    history.messages[-10:]
                    if hasattr(history, "messages")
                    else history[-10:]
                ), 
            }
        )

        # Save to memory (store just the answer text)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response.answer))

        return response

    async def ask(
        self, question: str, session_id: str = "default", use_advanced: bool = True           
    ) -> str:
        """Ask a question against the research documents."""

        history = self._get_session_history(session_id)

        # Use basic or advanced retriever
        retriever = self._build_retriever(use_advanced=use_advanced)
        #docs=retriever.invoke(question)
        docs = await retriever.ainvoke(question)
        context = self._format_docs_for_context(docs)

        # Build the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Answer questions
    based ONLY on the provided context documents.

    Rules:
    1. Only use information from the context below
    2. If the context doesn't have the answer, say so
    3. Cite which sources you used (e.g. "According to Source 1...")
    4. Rate your confidence: high, medium, or low""",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Question: {question}

    Provide a clear answer with source citations.""",
                ),
            ]
        )

        # Build and run the chain
        chain = prompt | self.llm | StrOutputParser()

        #response = chain.invoke(
        response = await chain.ainvoke(    
            {
                "context": context,
                "question": question,
                "history": history.messages[-10:],  # Last 10 messages for context
            }
        )

        # save Q&A to history
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response))

        return response

    def clear_session(self, session_id: str):
        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared session: {session_id}")

    def get_session_messages(self, session_id: str) -> list:
        """Get conversation history as dictionaries."""
        if session_id not in self.session_store:
            return []
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in self.session_store[session_id].messages
        ]

    def compare_retrievers(self, question: str):
        """Comparison of retrieval - basic vs advanced."""

        print(f'Question: "{question}"\n')

        # --- Basic ---
        basic = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}     
        )
        basic_docs = basic.invoke(question)

        print("=" * 60)
        print(f"BASIC RETRIEVER: {len(basic_docs)} chunks")
        print("=" * 60)        

        basic_total_chars = 0
        for i, doc in enumerate(basic_docs):
            source = doc.metadata.get("source", "Unknown")
            basic_total_chars += len(doc.page_content)    
            print(f"\n  Chunk {i+1} [{source}] ({len(doc.page_content)} chars):")
            print(f"  {doc.page_content[:150]}...")

        print(f"\n  Total text sent to LLM: {basic_total_chars} chars")

        # --- Advanced ---
        advanced = self._build_retriever(use_advanced=True)
        advanced_docs = advanced.invoke(question)

        print("\n" + "=" * 60)
        print(f"ADVANCED RETRIEVER: {len(advanced_docs)} chunks")
        print("=" * 60)

        advanced_total_chars = 0
        for i, doc in enumerate(advanced_docs):
            source = doc.metadata.get("source", "Unknown")
            advanced_total_chars += len(doc.page_content)
            print(f"\n  Chunk {i+1} [{source}] ({len(doc.page_content)} chars):")
            print(f"  {doc.page_content[:150]}...")

        print(f"\n  Total text sent to LLM: {advanced_total_chars} chars")

        # --- Summary ---
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"  Basic:    {len(basic_docs)} chunks, {basic_total_chars} chars")
        print(f"  Advanced: {len(advanced_docs)} chunks, {advanced_total_chars} chars")

        if advanced_total_chars < basic_total_chars:
            reduction = round((1 - advanced_total_chars / basic_total_chars) * 100)
            print(f"  Compression saved {reduction}% of tokens!")
        else:
            print(f"  Advanced found more targeted content")

def print_research_response(question: str, response: ResearchResponse):
    """Pretty print a structured research response."""

    print(f"\nQuestion: {question}")
    print(f"\n  Answer: {response.answer}")
    print(f"\n  Confidence: {response.confidence}")
    print(f"  Sources: {', '.join(response.sources)}")        

    if response.key_quotes:
        print(f"\n  Key Quotes:")
        for q in response.key_quotes:
            print(f'    - "{q}"')        

    print(f"\n  Follow-up Questions:")
    for fq in response.follow_up_questions:
        print(f"    - {fq}")
                                




















