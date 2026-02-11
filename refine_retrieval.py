"""
Retrieval Refinement Strategy for Corrective RAG
This implementation demonstrates a strategy that decomposes retrieved context into sentences,
filters them for relevance, and recomposes only the relevant information for answer generation.
"""

from typing import List, TypedDict
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DOCUMENT LOADING AND INDEXING
# =============================================================================

def load_and_index_documents(pdf_paths: List[str]):
    """Load PDF documents, split into chunks, and create vector store."""
    # Load documents from multiple PDFs
    docs = []
    for pdf_path in pdf_paths:
        docs.extend(PyPDFLoader(pdf_path).load())
    
    # Split documents into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, 
        chunk_overlap=150
    ).split_documents(docs)
    
    # Clean up encoding issues
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    
    # Create vector store and retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    return retriever


# =============================================================================
# STATE DEFINITION
# =============================================================================

class State(TypedDict):
    question: str
    docs: List[Document]
    
    strips: List[str]            # output of decomposition (sentence strips)
    kept_strips: List[str]       # after filtering (kept sentences)
    refined_context: str         # recomposed internal knowledge (joined kept_strips)
    
    answer: str


# =============================================================================
# SENTENCE DECOMPOSER
# =============================================================================

def decompose_to_sentences(text: str) -> List[str]:
    """Break text into individual sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# =============================================================================
# RELEVANCE FILTER
# =============================================================================

class KeepOrDrop(BaseModel):
    """Model for LLM-based relevance judgment."""
    keep: bool


def create_filter_chain(llm):
    """Create a chain to judge sentence relevance."""
    filter_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only."
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ])
    
    return filter_prompt | llm.with_structured_output(KeepOrDrop)


# =============================================================================
# GRAPH NODES
# =============================================================================

def create_retrieve_node(retriever):
    """Create the retrieval node."""
    def retrieve(state: State) -> State:
        q = state["question"]
        return {"docs": retriever.invoke(q)}
    return retrieve


def create_refine_node(filter_chain):
    """Create the refinement node (decompose -> filter -> recompose)."""
    def refine(state: State) -> State:
        q = state["question"]
        
        # Combine retrieved docs into one context string
        context = "\n\n".join(d.page_content for d in state["docs"]).strip()
        
        # 1) DECOMPOSITION: context -> sentence strips
        strips = decompose_to_sentences(context)
        
        # 2) FILTER: keep only relevant strips
        kept: List[str] = []
        for s in strips:
            if filter_chain.invoke({"question": q, "sentence": s}).keep:
                kept.append(s)
        
        # 3) RECOMPOSE: glue kept strips back together (internal knowledge)
        refined_context = "\n".join(kept).strip()
        
        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": refined_context,
        }
    return refine


def create_generate_node(llm):
    """Create the answer generation node."""
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say: 'I don't know based on the provided books.'"
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ])
    
    def generate(state: State) -> State:
        out = (answer_prompt | llm).invoke({
            "question": state["question"], 
            "refined_context": state['refined_context']
        })
        return {"answer": out.content}
    
    return generate


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_retrieval_refinement_graph(retriever, llm):
    """Build the complete retrieval refinement graph."""
    # Create filter chain
    filter_chain = create_filter_chain(llm)
    
    # Create nodes
    retrieve = create_retrieve_node(retriever)
    refine = create_refine_node(filter_chain)
    generate = create_generate_node(llm)
    
    # Build graph
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("refine", refine)
    g.add_node("generate", generate)
    
    # Connect nodes
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)
    
    return g.compile()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Load and index documents
    pdf_paths = [
        "./documents/book1.pdf",
        "./documents/book2.pdf",
        "./documents/book3.pdf"
    ]
    retriever = load_and_index_documents(pdf_paths)
    
    # Build graph
    app = build_retrieval_refinement_graph(retriever, llm)
    
    # Example query
    question = "Explain the bias–variance tradeoff"
    
    # Run the graph
    result = app.invoke({
        "question": question,
        "docs": [],
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "answer": ""
    })
    
    # Print results
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print("\n" + "=" * 80)
    print("ANSWER:")
    print(result["answer"])
    print("\n" + "=" * 80)
    print(f"Retrieved {len(result['docs'])} documents")
    print(f"Decomposed into {len(result['strips'])} sentences")
    print(f"Kept {len(result['kept_strips'])} relevant sentences")
    print("=" * 80)


if __name__ == "__main__":
    main()
