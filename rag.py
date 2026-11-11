import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from groq import Groq
import pymupdf as fitz
from collections import Counter
 
BULLET_CHARS = {"•", "◦", "▪"}

def detect_heading_sizes(doc, min_heading_ratio=0.15):
    """
    Automatically figure out this PDF's font-size structure -- no manual
    font-size survey needed for each new book. Pure statistics: most
    frequent size = body text; rarer, bigger sizes = heading tiers;
    a mostly-pure-digit size = page numbers.
    """
    size_counts = Counter()
    size_digit_lines = Counter()
    size_total_lines = Counter()

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                size = round(spans[0]["size"], 1)
                size_counts[size] += 1
                size_total_lines[size] += 1
                if text.isdigit():
                    size_digit_lines[size] += 1

    if not size_counts:
        raise ValueError("No text found in PDF -- is it scanned/image-based?")

    body_size, body_count = size_counts.most_common(1)[0]

    heading_sizes = [
        size for size, count in size_counts.items()
        if size >= body_size * 1.15
        and count <= body_count * min_heading_ratio
    ]
    heading_sizes.sort(reverse=True)

    page_number_size = None
    for size, total in size_total_lines.items():
        if size == body_size:
            continue
        digit_ratio = size_digit_lines[size] / total
        if digit_ratio > 0.8 and total > 3:
            page_number_size = size
            break

    return {
        "body_size": body_size,
        "heading_sizes": heading_sizes,
        "page_number_size": page_number_size,
    }
 
def get_lines_with_style(doc):
    """
    Walk every page and pull out each line of text along with:
      - size: font size (points) -- our signal for heading vs body text
      - bold: whether it's bold
      - x0, y0: position on the page -- needed to fix reading order
    get_text("dict") gives us this structural info; get_text("text") throws
    it all away and just gives raw strings.
    """
    lines_info = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
 
        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue  # image blocks etc. have no text lines, skip them
 
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
 
                text = "".join(span["text"] for span in spans).strip()
                if not text:
                    continue
 
                size = round(spans[0]["size"], 1)
                font_name = spans[0]["font"]
                is_bold = "Bold" in font_name or "bold" in font_name
                x0, y0 = line["bbox"][0], line["bbox"][1]
 
                lines_info.append({
                    "page": page_num + 1,
                    "text": text,
                    "size": size,
                    "bold": is_bold,
                    "x0": x0,
                    "y0": y0,
                })
 
    return lines_info
 
 
def sort_lines_reading_order(lines_info):
    """
    Sort lines top-to-bottom, left-to-right. This fixes the bug where a
    bullet symbol showed up AFTER its text -- PyMuPDF's raw extraction
    order doesn't always match visual layout.
    """
    def sort_key(line):
        y_bucket = round(line["y0"] / 3)
        return (line["page"], y_bucket, line["x0"])
 
    return sorted(lines_info, key=sort_key)
 
 
def remove_page_number_lines(lines_info, page_number_size):
    """
    Drop lines that are just a printed page number. page_number_size
    now comes from detect_heading_sizes() instead of being hardcoded.
    """
    if page_number_size is None:
        return lines_info
    return [
        line for line in lines_info
        if not (line["size"] == page_number_size and line["text"].strip().isdigit())
    ]
 
def is_bullet_line(text):
    """True if the line starts with (or is only) a bullet symbol."""
    stripped = text.strip()
    return stripped in BULLET_CHARS or (
        len(stripped) > 1 and stripped[0] in BULLET_CHARS
    )
 
 
def classify_and_group(lines_info, heading_sizes):
    """
    Walk the correctly-ordered lines and group them into sections
    (heading + its bullets). Uses exact heading font sizes found via the
    font-size survey we ran on this specific book: 24.0 (book title),
    18.8 (chapter), 15.0 (section). Everything else (11.2) is body text.
    """
    HEADING_SIZES = set(heading_sizes)
 
    sections = []
    current_heading = None
    current_heading_size = None
    current_bullets = []
    current_page = None
    pending_bullet = False
 
    for line in lines_info:
        text = line["text"]
        size = line["size"]
 
        is_heading = size in HEADING_SIZES
 
        if is_bullet_line(text):
            clean_text = text.lstrip("".join(BULLET_CHARS)).strip()
            if clean_text:
                current_bullets.append((line["x0"], clean_text))
                pending_bullet = False
            else:
                pending_bullet = True
            continue
 
        if pending_bullet:
            current_bullets.append((line["x0"], text))
            pending_bullet = False
        elif is_heading:
            if size == current_heading_size and not current_bullets:
                # wrapped 2nd line of the SAME heading
                current_heading += " " + text
            else:
                # genuinely new heading -> close previous section
                if current_heading is not None:
                    sections.append({
                        "heading": current_heading,
                        "bullets": current_bullets,
                        "page": current_page,
                    })
                current_heading = text
                current_heading_size = size
                current_bullets = []
                current_page = line["page"]
        else:
            # wrapped continuation OR a plain prose paragraph w/ no bullet
            if current_bullets:
                last_x0, last_text = current_bullets[-1]
                current_bullets[-1] = (last_x0, last_text + " " + text)
            elif current_heading is not None:
                current_bullets.append((line["x0"], text))
            else:
                current_heading = text
                current_heading_size = size
                current_page = line["page"]
 
    if current_heading is not None:
        sections.append({
            "heading": current_heading,
            "bullets": current_bullets,
            "page": current_page,
        })
 
    return sections
 
 
def sections_to_chunks(sections, max_chars=1000, overlap_chars=150):
    """
    Format each section as heading + '- bullet' lines -> one or more
    chunks. For sections too long to fit, hand off to LangChain's
    RecursiveCharacterTextSplitter for splitting + overlap.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n- ", "\n", ". ", " ", ""],
    )
 
    chunks = []
    for s in sections:
        heading = s["heading"]
        bullets = [text for _, text in s["bullets"]]
        page = s["page"]
 
        if not bullets:
            chunks.append({"page": page, "heading": heading, "text": heading})
            continue
 
        full_text = heading + "\n" + "\n".join(f"- {b}" for b in bullets)
 
        if len(full_text) <= max_chars:
            chunks.append({"page": page, "heading": heading, "text": full_text})
            continue
 
        pieces = splitter.split_text(full_text)
        for piece in pieces:
            chunks.append({"page": page, "heading": heading, "text": piece})
 
    return chunks
 
 
def extract_structured_chunks(pdf_path):
    """Main entry point: PDF path in -> list of structured chunks out."""
    doc = fitz.open(pdf_path)

    detected = detect_heading_sizes(doc)
    print(f"Detected body_size={detected['body_size']}, "
          f"heading_sizes={detected['heading_sizes']}, "
          f"page_number_size={detected['page_number_size']}")

    lines_info = get_lines_with_style(doc)
    doc.close()

    lines_info = sort_lines_reading_order(lines_info)
    lines_info = remove_page_number_lines(lines_info, detected["page_number_size"])
    sections = classify_and_group(lines_info, detected["heading_sizes"])
    chunks = sections_to_chunks(sections)

    print(f"Created {len(chunks)} chunks")
    return chunks
 
def print_chunk_overview(chunks):
    """Quick scan: index, page, char length, heading."""
    for i, c in enumerate(chunks):
        print(f"[{i}] page {c['page']} | {len(c['text'])} chars | {c['heading'][:50]}")
 
 
def save_chunks(chunks, path="chunks.json"):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks -> {path}")
 
 
def load_chunks(path="chunks.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_pipeline(pdf_path):
    chunks = extract_structured_chunks(pdf_path)
    print_chunk_overview(chunks)
    save_chunks(chunks)

pdf_path = "E:\\rag_eval_pipeline\\AI-Engineering-Building-Applications-with-Foundation-Models.pdf"
#run_pipeline(pdf_path)
from sentence_transformers import SentenceTransformer
import chromadb

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    if not texts:
        return []
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def build_vector_store(chunks, persist_dir="chroma_db", collection_name="book_chunks", force_rebuild=False):
    client = chromadb.PersistentClient(path=persist_dir)

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        if not force_rebuild:
            collection = client.get_collection(name=collection_name)
            print(f"Reusing existing Chroma collection '{collection_name}' with stored embeddings")
            return collection
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c["page"], "heading": c["heading"]} for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"Stored {len(chunks)} chunks in Chroma collection '{collection_name}'")
    return collection
 
 
def query_store(collection, question, n_results=3):
    # Question embedding is generated per query, but the document embeddings are
    # already stored and reused from Chroma. The model is cached at module level,
    # so it is not re-created repeatedly for each call.
    model = get_embedding_model()
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
 
    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        output.append({
            "text": doc,
            "page": meta["page"],
            "heading": meta["heading"],
            "distance": dist,
        })
    return output
 
 
def run_query_example():
    chunks = load_chunks("chunks.json")
    collection = build_vector_store(chunks)
    results = query_store(collection, "What is RAG?", n_results=3)
    for r in results:
        print(f"[page {r['page']}] {r['heading']} (distance={r['distance']:.3f})")
        print(r['text'][:200])
        print("---")

#run_query_example()

 
_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=get_groq_api_key())
    return _groq_client
 
 
def build_prompt(question, retrieved_chunks):
    """
    Combine retrieved chunks into a context block, then instruct the model
    to answer ONLY from this context -- this is what makes it RAG instead
    of just asking the model directly.
    """
    context_parts = []
    for c in retrieved_chunks:
        context_parts.append(f"[Page {c['page']} - {c['heading']}]\n{c['text']}")
    context = "\n\n".join(context_parts)
 
    prompt = f"""You are answering questions about a specific book, using ONLY the excerpts provided below.
 
Rules:
- Answer using ONLY the information in the excerpts below. Do NOT use any outside knowledge or state facts not directly written in the text.
- If the excerpts don't contain enough information to answer, say so clearly instead of guessing or extrapolating.
- Omit any details or generalizations that are not explicitly mentioned in the provided excerpts.
- When possible, mention which page(s) your answer comes from (e.g. [Page 32]).
 
Book excerpts:
{context}
 
Question: {question}
 
Answer:"""
    return prompt
 
 
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_model = None


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _reranker_model


def query_store_with_reranker(collection, question, initial_n=10, top_k=3):
    """
    Retrieve a larger initial pool (initial_n) from ChromaDB using vector similarity,
    then re-rank them with a CrossEncoder model to produce the top_k most relevant chunks.
    """
    candidates = query_store(collection, question, n_results=initial_n)
    if not candidates:
        return []
    
    reranker = get_reranker_model()
    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)
        
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def ask_question(collection, question, n_results=3, use_reranker=False, initial_n=10, verbose=True):
    """Full RAG flow: retrieve relevant chunks (with optional cross-encoder reranking), then generate an answer.
    Returns (answer_text, retrieved_chunks)."""
    if use_reranker:
        retrieved = query_store_with_reranker(collection, question, initial_n=initial_n, top_k=n_results)
    else:
        retrieved = query_store(collection, question, n_results=n_results)
        
    prompt = build_prompt(question, retrieved)

    client = get_groq_client()
    response = client.chat.completions.create(
        # NOTE: Groq deprecates models periodically -- check
        # console.groq.com/docs/models if this ever errors as decommissioned
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    if verbose:
        print("--- Answer ---")
        print(response.choices[0].message.content)
        print("\n--- Chunks used ---")
        for i, c in enumerate(retrieved):
            score_str = f" | rerank_score={c['rerank_score']:.3f}" if "rerank_score" in c else f" | dist={c['distance']:.3f}"
            print(f"[{i+1}] page {c['page']} | {c['heading']}{score_str}")

    return response.choices[0].message.content, retrieved 



load_dotenv()


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Add it to your .env file or export it in your shell."
        )
    return value


def get_groq_api_key() -> str:
    return require_env("GROQ_API_KEY")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BOOK_REGISTRY = {
    "ml_book": {
        "id": "ml_book",
        "title": "The Little Book of Deep Learning",
        "author": "François Fleuret",
        "description": "Foundations, models, training protocols, skip connections & compute schism.",
        "pdf_path": os.path.join(BASE_DIR, "ml_pdf.pdf"),
        "collection_name": "book_chunks",
        "chunks_path": os.path.join(BASE_DIR, "chunks.json"),
        "icon": "📖"
    },
    "ai_engineering_book": {
        "id": "ai_engineering_book",
        "title": "AI Engineering: Building Applications with Foundation Models",
        "author": "Chip Huyen",
        "description": "Building AI applications, foundation models, evaluation & production patterns.",
        "pdf_path": os.path.join(BASE_DIR, "AI-Engineering-Building-Applications-with-Foundation-Models.pdf"),
        "collection_name": "ai_engineering_book",
        "chunks_path": os.path.join(BASE_DIR, "ai_engineering_chunks.json"),
        "icon": "🤖"
    }
}


def get_registered_books():
    """Returns list of metadata for all registered books for the UI."""
    return [
        {
            "id": info["id"],
            "title": info["title"],
            "author": info["author"],
            "description": info["description"],
            "icon": info["icon"],
        }
        for info in BOOK_REGISTRY.values()
    ]


def get_vector_store_for_book(book_id="ml_book", force_rebuild=False):
    """Retrieve or create Chroma collection for a specific registered book."""
    if book_id not in BOOK_REGISTRY:
        raise ValueError(f"Unknown book_id '{book_id}'. Registered: {list(BOOK_REGISTRY.keys())}")
    
    info = BOOK_REGISTRY[book_id]
    return get_or_create_vector_store(
        pdf_path=info["pdf_path"],
        persist_dir="chroma_db",
        collection_name=info["collection_name"],
        chunks_path=info["chunks_path"],
        force_rebuild=force_rebuild
    )


def query_book_rag(book_id, question, n_results=5, use_reranker=True, initial_n=15):
    """
    RAG query targeted to a specific book context.
    Returns dict with answer, book info, and sources.
    """
    if book_id not in BOOK_REGISTRY:
        book_id = "ml_book"
    
    book_info = BOOK_REGISTRY[book_id]
    collection = get_vector_store_for_book(book_id, force_rebuild=False)
    answer, retrieved = ask_question(collection, question, n_results=n_results, use_reranker=use_reranker, initial_n=initial_n, verbose=False)
    
    sources = []
    for c in retrieved:
        src = {
            "page": c["page"],
            "heading": c["heading"],
            "text": c["text"]
        }
        if "rerank_score" in c:
            src["rerank_score"] = round(c["rerank_score"], 3)
        elif "distance" in c:
            src["distance"] = round(c["distance"], 3)
        sources.append(src)
        
    return {
        "book_id": book_id,
        "book_title": book_info["title"],
        "book_icon": book_info["icon"],
        "answer": answer,
        "sources": sources
    }


def ask_question_with_rag(question, pdf_path=None, n_results=3, use_reranker=True, initial_n=10):
    """
    High-level function to ask a question using RAG.
    Reuses pre-stored ChromaDB vector embeddings directly without re-parsing PDF or re-generating chunk embeddings.
    """
    if pdf_path is None:
        pdf_path = os.path.join(BASE_DIR, "ml_pdf.pdf")
    collection = get_or_create_vector_store(pdf_path, force_rebuild=False)
    return ask_question(collection, question, n_results=n_results, use_reranker=use_reranker, initial_n=initial_n)

#ask_question_with_rag("What is Perplexity?", n_results=3)




def get_ragas_judge():
    """Set up the judge LLM (Groq) + embeddings (local) once, reused
    across all metrics/questions."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings

    judge_client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    judge_llm = llm_factory(
        "openai/gpt-oss-120b",
        client=judge_client,
        max_tokens=4096,
    )
    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")
    return judge_llm, embeddings
 
 
def run_evaluation(collection, test_set, n_results=3, use_reranker=False, initial_n=10):
    """
    For each question: retrieve + generate (our actual pipeline), then
    score the result with 3 metrics -- no reference/ground-truth needed:
      - faithfulness:      does the answer stick to the retrieved context?
      - answer_relevancy:  does the answer address the question asked?
      - context_precision: were the retrieved chunks actually relevant?
    Returns a list of per-question score dicts.
    """
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference

    judge_llm, embeddings = get_ragas_judge()
 
    faithfulness_metric = Faithfulness(llm=judge_llm)
    relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)
    precision_metric = ContextPrecisionWithoutReference(llm=judge_llm)
 
    all_results = []
    for item in test_set:
        question = item["question"]
        answer, retrieved = ask_question(
            collection, question, n_results=n_results, use_reranker=use_reranker, initial_n=initial_n, verbose=False
        )
        contexts = [r["text"] for r in retrieved]
 
        faith_score = faithfulness_metric.score(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        rel_score = relevancy_metric.score(user_input=question, response=answer)
        prec_score = precision_metric.score(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
 
        all_results.append({
            "question": question,
            "answer": answer,
            "sources": [
                {"page": chunk["page"], "heading": chunk["heading"]}
                for chunk in retrieved
            ],
            "faithfulness": faith_score.value,
            "answer_relevancy": rel_score.value,
            "context_precision": prec_score.value,
        })
 
    return all_results
 
 
def print_evaluation_results(results):
    """Print readable answers, scores, sources, and averages."""
    if not results:
        print("No evaluation results.")
        return

    print("\n" + "=" * 72)
    print("RAG EVALUATION RESULTS")
    print("=" * 72)

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] QUESTION")
        print(result["question"])
        print("\nANSWER")
        print(result["answer"].strip())
        print("\nSCORES")
        print(f"  Faithfulness:      {result['faithfulness']:.2f}")
        print(f"  Answer relevancy:  {result['answer_relevancy']:.2f}")
        print(f"  Context precision: {result['context_precision']:.2f}")
        print("\nSOURCES")
        for source in result["sources"]:
            print(f"  Page {source['page']}: {source['heading']}")
 
    n = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_rel = sum(r["answer_relevancy"] for r in results) / n
    avg_prec = sum(r["context_precision"] for r in results) / n
    print("\n" + "=" * 72)
    print("AVERAGE SCORES")
    print("=" * 72)
    print(f"Faithfulness:      {avg_faith:.2f}")
    print(f"Answer relevancy:  {avg_rel:.2f}")
    print(f"Context precision: {avg_prec:.2f}")


def get_or_create_vector_store(pdf_path, persist_dir="chroma_db", collection_name="book_chunks", chunks_path="chunks.json", force_rebuild=False):
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [c.name for c in client.list_collections()]

    # Check if embeddings are already stored on disk
    if collection_name in existing and not force_rebuild:
        collection = client.get_collection(name=collection_name)
        if collection.count() > 0:
            print(f"Loaded existing ChromaDB collection '{collection_name}' ({collection.count()} chunks). Skipping PDF parsing and embedding generation!")
            return collection

    # If collection is missing or force_rebuild is True:
    if os.path.exists(chunks_path):
        print(f"Loading pre-extracted chunks from: {chunks_path}")
        chunks = load_chunks(chunks_path)
    elif os.path.exists(pdf_path):
        print(f"Initializing vector store. Extracting chunks from: {pdf_path}")
        chunks = extract_structured_chunks(pdf_path)
        save_chunks(chunks, path=chunks_path)
    else:
        raise FileNotFoundError(f"Neither chunks file '{chunks_path}' nor PDF '{pdf_path}' found.")
    return build_vector_store(chunks, persist_dir=persist_dir, collection_name=collection_name, force_rebuild=True)



if __name__ == "__main__":
    pdf_path = os.path.join(BASE_DIR, "ml_pdf.pdf")
    print(f"Extracting chunks from: {pdf_path}")
    chunks = extract_structured_chunks(pdf_path)
    print_chunk_overview(chunks)
    
    # Store chunks to JSON file
    save_chunks(chunks, path="chunks.json")
    
    # Embed chunks and store in ChromaDB vector store
    print("\nLoading / updating ChromaDB vector store...")
    collection = build_vector_store(chunks, persist_dir="chroma_db", collection_name="book_chunks", force_rebuild=False)
    print("\nSuccessfully loaded vector store from ChromaDB!")

    # Define reduced test set for RAG evaluation to save LLM cost
    test_set = [
        {"question": "What are skip connections and why are they used in deep architectures?"},
        {"question": "How do GPUs, TPUs, and batching contribute to efficient computation in deep learning?"},
    ]

    print("\nEvaluating RAG pipeline performance (Top-5 Reranked Chunks) using Ragas...")
    results_reranked = run_evaluation(collection, test_set, n_results=5, use_reranker=True, initial_n=15)
    print_evaluation_results(results_reranked)