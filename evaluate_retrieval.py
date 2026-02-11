"""
Retrieval Evaluator - Corrective RAG Strategy

This implements a retrieval evaluation approach for RAG systems that:
1. Retrieves relevant documents from a vector store
2. Evaluates each document with a score-based LLM judge
3. Routes based on evaluation:
   - CORRECT: At least one doc scores > upper threshold → refine and generate
   - INCORRECT: All docs score < lower threshold → trigger web search/fail
   - AMBIGUOUS: Mixed scores → handle ambiguous case
4. For CORRECT case: decomposes, filters, and recomposes context
5. Generates answer using refined context
"""

from typing import List, TypedDict, Literal
from pydantic import BaseModel
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================
# Configuration
# =============================

UPPER_TH = 0.7  # Threshold for "good" documents
LOWER_TH = 0.3  # Threshold for "bad" documents


# =============================
# Document Loading and Indexing
# =============================

def load_and_index_documents(pdf_paths: List[str]):
    """Load PDFs, chunk them, and create a vector store."""
    # Load documents
    docs = []
    for pdf_path in pdf_paths:
        docs.extend(PyPDFLoader(pdf_path).load())
    
    # Split into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, 
        chunk_overlap=150
    ).split_documents(docs)
    
    # Clean encoding
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    return retriever


# =============================
# State Definition
# =============================

class State(TypedDict):
    question: str
    docs: List[Document]
    
    good_docs: List[Document]
    verdict: str
    reason: str
    
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    
    answer: str


# =============================
# Document Evaluation
# =============================

class DocEvalScore(BaseModel):
    score: float
    reason: str


def create_doc_eval_chain(llm):
    """Create a chain to evaluate document relevance with scores."""
    doc_eval_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only."
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ])
    
    return doc_eval_prompt | llm.with_structured_output(DocEvalScore)


# =============================
# Sentence Decomposition and Filtering
# =============================

def decompose_to_sentences(text: str) -> List[str]:
    """Decompose text into individual sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool


def create_filter_chain(llm):
    """Create a chain to filter sentences for relevance."""
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


# =============================
# Graph Nodes
# =============================

def retrieve_node(state: State, retriever) -> State:
    """Retrieve relevant documents."""
    q = state["question"]
    return {"docs": retriever.invoke(q)}


def eval_each_doc_node(state: State, doc_eval_chain) -> State:
    """Evaluate each retrieved document and determine verdict."""
    q = state["question"]
    
    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []
    
    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)
        
        # Keep docs with score > LOWER_TH for refinement
        if out.score > LOWER_TH:
            good.append(d)
    
    # CORRECT if at least one doc > UPPER_TH
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }
    
    # INCORRECT if all docs < LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        why = "No chunk was sufficient."
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. {why}",
        }
    
    # AMBIGUOUS - anything in between
    why = "Mixed relevance signals."
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }


def refine(state: State, filter_chain) -> State:
    """Refine context using decompose -> filter -> recompose."""
    q = state["question"]
    
    # Combine good docs (those with score > LOWER_TH)
    context = "\n\n".join(d.page_content for d in state["good_docs"]).strip()
    
    # 1) DECOMPOSITION: context -> sentence strips
    strips = decompose_to_sentences(context)
    
    # 2) FILTER: keep only relevant strips
    kept: List[str] = []
    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s)
    
    # 3) RECOMPOSE: glue kept strips back together
    refined_context = "\n".join(kept).strip()
    
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }


def generate(state: State, llm) -> State:
    """Generate answer using refined context."""
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'"
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ])
    
    out = (answer_prompt | llm).invoke({
        "question": state["question"], 
        "refined_context": state["refined_context"]
    })
    
    return {"answer": out.content}


def fail_node(state: State) -> State:
    """Handle INCORRECT verdict - would trigger web search in production."""
    return {"answer": f"FAIL: {state['reason']}"}


def ambiguous_node(state: State) -> State:
    """Handle AMBIGUOUS verdict."""
    return {"answer": f"Ambiguous: {state['reason']}"}


def route_after_eval(state: State) -> str:
    """Route based on evaluation verdict."""
    if state["verdict"] == "CORRECT":
        return "refine"
    elif state["verdict"] == "INCORRECT":
        return "web_search"
    else:
        return "ambiguous"


# =============================
# Graph Construction
# =============================

def build_retrieval_evaluator_graph(retriever, llm):
    """Build the corrective RAG graph with retrieval evaluation."""
    # Create evaluation and filter chains
    doc_eval_chain = create_doc_eval_chain(llm)
    filter_chain = create_filter_chain(llm)
    
    # Build graph
    g = StateGraph(State)
    
    # Add nodes with bound dependencies
    g.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    g.add_node("eval_each_doc", lambda state: eval_each_doc_node(state, doc_eval_chain))
    g.add_node("refine", lambda state: refine(state, filter_chain))
    g.add_node("generate", lambda state: generate(state, llm))
    g.add_node("fail", fail_node)
    g.add_node("ambiguous", ambiguous_node)
    
    # Add edges
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    
    # Conditional routing after evaluation
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {"refine": "refine", "web_search": "fail", "ambiguous": "ambiguous"}
    )
    
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)
    g.add_edge("fail", END)
    g.add_edge("ambiguous", END)
    
    return g.compile()


# =============================
# Main Execution
# =============================

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
    app = build_retrieval_evaluator_graph(retriever, llm)
    
    # Example queries
    questions = [
        "Explain the bias-variance tradeoff",
        "What are attention mechanisms and why are they important in current models?",
        "AI news from last week"
    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}\n")
        
        result = app.invoke({
            "question": question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "answer": "",
        })
        
        print(f"VERDICT: {result['verdict']}")
        print(f"REASON: {result['reason']}")
        print(f"\nOUTPUT:\n{result['answer']}")
        print()


if __name__ == "__main__":
    main()
