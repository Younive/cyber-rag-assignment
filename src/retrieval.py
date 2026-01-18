from vectorstore.manage_vectorstore import VectorStoreManager
from prompt_template import build_gemini_rag_prompt
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize vectorstore once
vectorstore = VectorStoreManager().get_exist_cromadb()

def retrieve_documents(query: str, k: int = 3):
    """
    Retrieve relevant documents from the vector store.
    
    Args:
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of Document objects with metadata
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

def retrieve_with_scores(query: str, k: int = 3):
    """
    Retrieve relevant documents with similarity scores.
    
    Args:
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of tuples (Document, score)
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

def get_rag_prompt(query: str, k: int = 3, language: str = "auto"):
    """
    Get complete RAG prompt ready for LLM inference.
    
    Args:
        query: User's question
        k: Number of documents to retrieve
        language: 'en', 'th', or 'auto'
        
    Returns:
        Complete prompt string with retrieved context
    """
    results = retrieve_documents(query, k=k)
    prompt = build_gemini_rag_prompt(query, results, language=language)
    return prompt, results

def main():
    """Test retrieval functionality"""
    print("\n" + "="*80)
    print("Testing Retrieval System")
    print("="*80)
    
    # Test query
    query = "How does MITRE describe the purpose of Persistence techniques?   "
    print(f"\nQuery: {query}")
    print("-"*80)
    
    # Retrieve documents
    results = retrieve_documents(query, k=3)
    
    print(f"\nFound {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Page: {doc.metadata.get('page', 'unknown')}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
    
    # Generate prompt
    print("\n" + "="*80)
    print("Generated RAG Prompt")
    print("="*80)
    prompt, _ = get_rag_prompt(query, k=3)
    print(prompt)
    print("\n" + "="*80)
    
    # Test with similarity scores
    print("\nTesting with similarity scores...")
    print("-"*80)
    results_with_scores = retrieve_with_scores(query, k=3)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"{i}. Similarity Score: {score:.4f}")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Page: {doc.metadata.get('page', 'unknown')}")
        print()

    # rag response
    model = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))
    response = model.invoke(prompt)
    print("\n" + "="*80)
    print("RAG Response")
    print("="*80)
    print(response)
    print("\n" + "="*80)

if __name__ == "__main__":
    main()