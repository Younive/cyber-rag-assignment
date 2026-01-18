import os
import shutil
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_knowledge_base"

class VectorStoreManager:
    """Manage ChromaDB vector store operations"""
    
    def __init__(
        self, 
        persist_directory: str = CHROMA_DB_PATH, 
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = "models/text-embedding-004"
    ):
        """
        Initialize VectorStoreManager.
        
        Args:
            persist_directory: Path to the ChromaDB persistence directory
            collection_name: Name of the collection (must match rag_pipeline.py)
            embedding_model_name: Name of the Google embedding model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # CRITICAL FIX: Initialize the embedding function as an object, not a string
        self.embedding_function = GoogleGenerativeAIEmbeddings(model=embedding_model_name, api_key=os.getenv("GOOGLE_API_KEY"))
        print(f"Initialized VectorStoreManager")
        print(f"  Collection: {collection_name}")
        print(f"  Embedding: {embedding_model_name}")
    
    def get_exist_cromadb(self):
        """
        Load existing ChromaDB vector store.
        
        Returns:
            Chroma vector store instance
            
        Raises:
            FileNotFoundError: If the persist directory doesn't exist
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"ChromaDB directory not found at {self.persist_directory}. "
                "Please run the RAG pipeline first to create the vector store."
            )
        
        print(f"Loading ChromaDB from {self.persist_directory}...")
        print(f"  Collection name: {self.collection_name}")
        
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,  # CRITICAL: Specify collection name
            embedding_function=self.embedding_function
        )
        
        # Verify vector store is not empty
        try:
            collection = vectorstore._collection
            count = collection.count()
            print(f"Successfully loaded vector store with {count} documents.")
            
            if count == 0:
                print("\nWARNING: Vector store is empty!")
                print("   This usually means:")
                print("   1. The collection name doesn't match rag_pipeline.py")
                print("   2. The RAG pipeline hasn't been run yet")
                print("   3. The vector store was cleared\n")
        except Exception as e:
            print(f"Warning: Could not verify vector store contents: {e}")
        
        return vectorstore
    
    def create_chromadb(self, documents):
        """
        Create a new ChromaDB vector store from documents.
        
        Args:
            documents: List of documents to add to the vector store
            
        Returns:
            Chroma vector store instance
        """
        # Delete existing vector store if it exists
        if self.check_chromadb_exists():
            print(f"Existing ChromaDB found. Deleting...")
            self.delete_chromadb()
        
        print(f"Creating new ChromaDB at {self.persist_directory}...")
        print(f"  Collection name: {self.collection_name}")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name  # CRITICAL: Specify collection name
        )
        
        print(f"ChromaDB created with {len(documents)} documents.")
        return vectorstore
    
    def delete_chromadb(self):
        """Delete existing ChromaDB vector store."""
        if os.path.exists(self.persist_directory):
            print(f"Deleting ChromaDB at {self.persist_directory}...")
            shutil.rmtree(self.persist_directory)
            print("ChromaDB deleted successfully.")
        else:
            print(f"No ChromaDB found at {self.persist_directory}.")
    
    def check_chromadb_exists(self) -> bool:
        """
        Check if ChromaDB vector store exists.
        
        Returns:
            True if exists, False otherwise
        """
        exists = os.path.exists(self.persist_directory)
        if exists:
            # Also check if it's not empty
            try:
                if os.path.isdir(self.persist_directory):
                    files = os.listdir(self.persist_directory)
                    return len(files) > 0
            except:
                pass
        return False
    
    def get_vectorstore_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if not self.check_chromadb_exists():
            return {"exists": False, "count": 0}
        
        try:
            vectorstore = self.get_exist_cromadb()
            collection = vectorstore._collection
            count = collection.count()
            
            # Get sample metadata if available
            sample = collection.peek(limit=10)
            sources = set()
            pages = set()
            
            if sample and 'metadatas' in sample and sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
                    if 'page' in metadata:
                        pages.add(str(metadata['page']))
            
            return {
                "exists": True,
                "count": count,
                "collection_name": self.collection_name,
                "sources": sorted(list(sources)),
                "sample_pages": sorted(list(pages))[:5],
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "exists": True,
                "count": 0,
                "error": str(e)
            }
    
    def add_documents(self, documents):
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Updated Chroma vector store instance
        """
        if not self.check_chromadb_exists():
            print("No existing vector store found. Creating new one...")
            return self.create_chromadb(documents)
        
        print(f"Adding {len(documents)} documents to existing vector store...")
        vectorstore = self.get_exist_cromadb()
        vectorstore.add_documents(documents)
        print(f"Added {len(documents)} documents successfully.")
        
        return vectorstore

if __name__ == "__main__":
    # Test the VectorStoreManager
    print("\n" + "="*80)
    print("VectorStoreManager Test")
    print("="*80 + "\n")
    
    manager = VectorStoreManager()
    
    stats = manager.get_vectorstore_stats()
    print(f"Vector Store Exists: {stats.get('exists', False)}")
    print(f"Collection Name: {stats.get('collection_name', 'N/A')}")
    print(f"Document Count: {stats.get('count', 0)}")
    
    if stats.get('sources'):
        print(f"\nSources in vector store:")
        for source in stats['sources']:
            print(f"  - {source}")
    
    if stats.get('sample_pages'):
        print(f"\nSample pages: {', '.join(stats['sample_pages'])}")
    
    if stats.get('error'):
        print(f"\n❌ Error: {stats['error']}")
    
    print("\n" + "="*80)