"""
Vector store and RAG functionality for contract analysis
"""

import os
import json
from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

class LegalVectorStore:
    def __init__(self, persist_directory: str = "data/vector_store"):
        """
        Initialize the legal vector store
        
        Args:
            persist_directory: Directory to store the vector store
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Load or create vector store
        if os.path.exists(persist_directory):
            self.vector_store = FAISS.load_local(persist_directory, self.embeddings)
        else:
            self.vector_store = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
        """
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
        else:
            self.vector_store.add_documents(texts)
        
        # Save the vector store
        self.vector_store.save_local(self.persist_directory)
    
    def load_legal_precedents(self, precedents_dir: str) -> None:
        """
        Load legal precedents from a directory
        
        Args:
            precedents_dir: Directory containing legal precedents
        """
        loader = DirectoryLoader(
            precedents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        self.add_documents(documents)
    
    def load_australian_law(self) -> None:
        """
        Load Australian legal principles and guidelines
        """
        # Load Australian Consumer Law
        acl_doc = Document(
            page_content=json.dumps({
                "title": "Australian Consumer Law",
                "content": """
                The Australian Consumer Law (ACL) provides automatic guarantees for goods and services.
                Key principles:
                - Unfair contract terms are void
                - Consumer guarantees cannot be excluded
                - Penalty clauses are unenforceable
                - Small business protections apply to contracts under $300,000
                """
            })
        )
        
        # Load ACCC Guidelines
        accc_doc = Document(
            page_content=json.dumps({
                "title": "ACCC Guidelines",
                "content": """
                ACCC guidelines on unfair contract terms:
                - Terms must be transparent
                - Notice periods should be balanced
                - Automatic renewals require clear notice
                - Late fees should be reasonable
                """
            })
        )
        
        self.add_documents([acl_doc, accc_doc])
    
    def search_relevant_precedents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant legal precedents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant precedents
        """
        if self.vector_store is None:
            return []
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Parse and return results
        results = []
        for doc in docs:
            try:
                content = json.loads(doc.page_content)
                results.append({
                    "title": content.get("title", ""),
                    "content": content.get("content", ""),
                    "score": doc.metadata.get("score", 0)
                })
            except json.JSONDecodeError:
                results.append({
                    "title": "Legal Precedent",
                    "content": doc.page_content,
                    "score": doc.metadata.get("score", 0)
                })
        
        return results
    
    def get_legal_context(self, clause_text: str) -> str:
        """
        Get relevant legal context for a clause
        
        Args:
            clause_text: The clause text to analyze
            
        Returns:
            Relevant legal context
        """
        # Search for relevant precedents
        precedents = self.search_relevant_precedents(clause_text)
        
        # Format the context
        context = "Relevant Legal Context:\n\n"
        for precedent in precedents:
            context += f"Title: {precedent['title']}\n"
            context += f"Content: {precedent['content']}\n\n"
        
        return context 