#=======================================
# 9.0. Full Production RAG Pipeline (FINAL)
#=======================================

import os
import re
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_anthropic import ChatAnthropic

#=======================================
# Setup
#=======================================

load_dotenv()
da_api_key_claude = os.getenv("DK_CLAUDE_API_KEY")
da_api_key_openai = os.getenv("DK_OPENAI_API_KEY")

embeddings_model = OpenAIEmbeddings(api_key=da_api_key_openai, model="text-embedding-3-small")

CHROMA_DB_PATH = r"/Users/dilip/Desktop/AppliedAI/RAG/dk_chroma_db"

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings_model,
    collection_name="support_tickets"
)


llm_brain = ChatAnthropic(
    anthropic_api_key=da_api_key_claude,
    #model="claude-sonnet-4-6",
    model="claude-haiku-4-5-20251001",
    max_tokens=200
)

print(f"Connected to vector store. Items indexed: {vectorstore._collection.count()}")

all_data = vectorstore.get()
chunk_texts = all_data["documents"]
chunk_metadatas = all_data["metadatas"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


tokenized_text = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_text)
print("BM25 index built.")

print("\nLoading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.\n")

TOP_K = 5
RETRIEVE_K = 10
CONFIDENCE_THRESHOLD = -6.0   # tune against your own cross-encoder score distribution


#=======================================
# STAGE 1: Intent-Based Routing (ALWAYS RUNS)
#=======================================

class RouteDecision(BaseModel):
    needs_retrieval: bool = Field(description="Does this question need document retrieval?")
    category: Optional[str] = Field(
        default=None,
        description="Ticket category if retrieval is needed (e.g. billing, account, security, onboarding, api)"
    )
    query_clarity: Literal["clear", "vague", "ambiguous"] = Field(
        description=(
            "clear: specific, well-formed query - search as-is. "
            "vague: short/casual phrasing with ONE likely meaning - needs rewriting. "
            "ambiguous: could genuinely mean SEVERAL different things - needs multi-query."
        )
    )

router_llm = llm_brain.with_structured_output(RouteDecision)


def route_query(raw_question):
    prompt = f"""Classify this support chat question.

Question: "{raw_question}"
"""
    return router_llm.invoke([HumanMessage(content=prompt)])


#=======================================
# STAGE 2: Retrieval Primitives (METADATA FILTERING + HYBRID SEARCH)
#=======================================

def dense_search(question, category=None, k=RETRIEVE_K):
    where = {"category": category} if category else None
    results = vectorstore.similarity_search_with_score(question, k=k, filter=where)
    return [(doc.metadata["source_id"], doc.metadata["title"], doc.page_content) for doc, score in results]


def sparse_search(question, category=None, k=RETRIEVE_K):
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(zip(chunk_texts, chunk_metadatas, scores))

    if category:
        scored_chunks = [c for c in scored_chunks if c[1]["category"] == category]

    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    top = scored_chunks[:k]
    return [(metadata["source_id"], metadata["title"], text) for text, metadata, score in top]


def reciprocal_rank_fusion(ranked_lists, k=60):
    rrf_scores = {}
    doc_info = {}
    for ranked_list in ranked_lists:
        for rank, (source_id, title, text) in enumerate(ranked_list, start=1):
            score = 1 / (k + rank)
            rrf_scores[source_id] = rrf_scores.get(source_id, 0) + score
            doc_info[source_id] = (title, text)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(source_id, doc_info[source_id][0], doc_info[source_id][1], score) for source_id, score in fused]


def rerank_fused_results(question, fused_results, top_n_to_rerank=8):
    candidates = fused_results[:top_n_to_rerank]
    pairs = [(question, text) for source_id, title, text, rrf_score in candidates]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [(source_id, title, ce_score) for (source_id, title, text, rrf_score), ce_score in reranked]


def hybrid_search(question, category=None, k=TOP_K):
    """Baseline: metadata-filtered hybrid search + rerank, single query."""
    dense_results = dense_search(question, category=category)
    sparse_results = sparse_search(question, category=category)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    reranked = rerank_fused_results(question, fused)
    return reranked[:k]


#=======================================
# STAGE 3a: Query Transformation (CONDITIONAL - "vague")
#=======================================

def rewrite_query(raw_question):
    prompt = f"""Rewrite this short, vague support question into a clear,
complete question. Do not answer it - only rewrite it. Respond with
ONLY the rewritten question.

Original question: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


#=======================================
# STAGE 3b: Multi-Query Retrieval (CONDITIONAL - "ambiguous")
#=======================================

def generate_query_variants(raw_question, n=3):
    prompt = f"""This question is ambiguous and could mean several things.
Generate {n} DIFFERENT, specific interpretations of what the user might
mean, covering different possible angles (e.g. permissions, performance/
errors, data export, account access, billing). Respond with ONLY the
{n} rewritten questions, one per line, nothing else.

Original question: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    variants = [line.strip("- ").strip() for line in response.content.strip().split("\n") if line.strip()]
    return variants[:n]


def multi_query_search(raw_question, category=None, n_variants=3, k=TOP_K):
    variants = generate_query_variants(raw_question, n=n_variants)
    all_variants = [raw_question] + variants

    per_variant_fused_lists = []
    for variant in all_variants:
        dense_results = dense_search(variant, category=category)
        sparse_results = sparse_search(variant, category=category)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        simplified = [(sid, title, text) for sid, title, text, score in fused]
        per_variant_fused_lists.append(simplified)

    final_fused = reciprocal_rank_fusion(per_variant_fused_lists)
    final_results = rerank_fused_results(raw_question, final_fused)
    return variants, final_results[:k]


#=======================================
# STAGE 4: Answer Generation (ALWAYS RUNS - grounded or direct)
#=======================================

def generate_answer(raw_question, results, is_ambiguous=False):
    """Grounded answer using retrieved ticket context."""
    if not results:
        return "I don't have enough information to answer that. Could you clarify your question?"

    context_blocks = []
    for source_id, title, score in results:
        for text, metadata in zip(chunk_texts, chunk_metadatas):
            if metadata["source_id"] == source_id:
                context_blocks.append(f"[{source_id}] {title}\n{text}")
                break

    context = "\n\n".join(context_blocks)

    ambiguity_note = (
        "This question could have multiple valid causes or meanings. "
        "Address EACH distinct scenario found in the context separately - "
        "do not pick just one and ignore the others.\n\n"
        if is_ambiguous else ""
    )

    prompt = f"""You are a support assistant. Answer the user's question
using ONLY the context below. {ambiguity_note}If the context doesn't fully
answer the question, say what you found and suggest contacting support for
more detail. Cite the ticket ID(s) you used, like [TCK-001].

Context:
{context}

Question: "{raw_question}"

Answer:"""

    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def generate_direct_answer(raw_question):
    """No retrieval needed - answer directly (e.g. greetings, chit-chat, opinion questions)."""
    prompt = f"""Answer this support chat message directly and briefly.
It does not require looking up any documentation.

Message: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


#=======================================
# ORCHESTRATOR: Full Production Pipeline
#=======================================

def production_pipeline(raw_question):
    print(f"\n{'='*70}\nQUESTION: \"{raw_question}\"\n{'='*70}")

    # Stage 1 — always: route
    decision = route_query(raw_question)
    print(f"Routing -> needs_retrieval={decision.needs_retrieval}, "
          f"category={decision.category}, clarity={decision.query_clarity}")

    if not decision.needs_retrieval:
        print("No retrieval needed - answering directly.")
        answer = generate_direct_answer(raw_question)
        print(f"\nGenerated Answer:\n{answer}")
        return None, answer

    # Only trust the category filter when the query is unambiguous.
    safe_category = decision.category if decision.query_clarity == "clear" else None
    is_ambiguous = decision.query_clarity == "ambiguous"

    # Stage 2 — always: hybrid search (filtered only if safe_category is set)
    results = hybrid_search(raw_question, category=safe_category)
    best_score = results[0][2] if results else float("-inf")

    # Stage 3 — conditional escalation
    if is_ambiguous:
        print("-> Ambiguous query: escalating to Multi-Query Retrieval (no category filter)")
        variants, results = multi_query_search(raw_question, category=safe_category)
        for i, v in enumerate(variants, start=1):
            print(f"   Variant {i}: \"{v}\"")

    elif decision.query_clarity == "vague" or best_score < CONFIDENCE_THRESHOLD:
        print("-> Vague query or low confidence: rewriting and re-searching (no category filter)")
        rewritten = rewrite_query(raw_question)
        print(f"   Rewritten: \"{rewritten}\"")
        results = hybrid_search(rewritten, category=safe_category)

    else:
        print("-> Clear query: using baseline hybrid search result (category filter applied)")

    print(f"\nFinal Top-{TOP_K}:")
    for rank, (source_id, title, score) in enumerate(results, start=1):
        print(f"  Rank {rank}: {source_id} | score={score:.4f} | {title}")

    # Stage 4 — always: generate the grounded final answer
    answer = generate_answer(raw_question, results, is_ambiguous=is_ambiguous)
    print(f"\nGenerated Answer:\n{answer}")

    return results, answer


#=======================================
# Test
#=======================================

test_cases = {
    "I can't access my reports": {
        "TCK-017": "Setting up custom user roles and permissions",
        "TCK-023": "Timeout errors on large report generation",
        "TCK-008": "Data export and account deletion",
    },
    "big file wont go through": {
        "TCK-013": "File upload size limits",
    },
    "What is your refund policy?": {
        "TCK-001": "Refund policy for Enterprise cancellations",
    },
}

for question, relevant_tickets in test_cases.items():
    print(f"\nRelevant tickets (any of these count as a good result): {list(relevant_tickets.keys())}")

    results, answer = production_pipeline(question)

    found = []
    if results:
        for source_id, title, score in results:
            if source_id in relevant_tickets:
                found.append(source_id)

    print(f"\nCoverage: {len(found)} / {len(relevant_tickets)} relevant tickets found -> {found}")