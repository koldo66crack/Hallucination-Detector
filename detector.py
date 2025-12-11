import os
import json
import nltk
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

load_dotenv()

# Ensure NLTK sentence splitter is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- Data Models (Structuring the OpenAI Output) ---
class AtomicClaim(BaseModel):
    claim: str = Field(..., description="A single, standalone factual statement.")
    original_sentence_reference: str = Field(..., description="The original sentence this claim came from.")


class DecompositionResponse(BaseModel):
    claims: List[AtomicClaim]


class HallucinationDetector:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Load the local embedding model (small & fast)
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

    def get_atomic_claims(self, text: str) -> List[AtomicClaim]:
        """Uses OpenAI Structured Outputs to break text into facts."""
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini", # Or gpt-4o-mini
            messages=[
                {"role": "system", "content": "You are a precise fact-checker."},
                {"role": "user", "content": f"Break this text into atomic claims that can be fact checked:\n\n{text}"}
            ],
            response_format=DecompositionResponse,
        )
        return completion.choices[0].message.parsed.claims

    def find_best_match(self, claim_text: str, retrieved_chunks: List[str]):
        """
        The 'Sliding Window' Logic.
        Matches a claim against every SENTENCE in every CHUNK.
        """
        claim_vector = self.encoder.encode([claim_text])
        
        best_score = -1.0
        best_evidence = ""
        best_chunk_index = -1

        for i, chunk in enumerate(retrieved_chunks):
            # 1. Split chunk into sentences (Granular Matching)
            sentences = nltk.sent_tokenize(chunk)
            if not sentences: continue
            
            # 2. Vectorize all sentences in this chunk at once
            chunk_vectors = self.encoder.encode(sentences)
            
            # 3. Calculate Cosine Similarity
            scores = cosine_similarity(claim_vector, chunk_vectors)[0]
            
            # 4. Find the max score in this chunk
            max_idx = np.argmax(scores)
            max_score = scores[max_idx]

            # 5. Global Max Tracking
            if max_score > best_score:
                best_score = float(max_score)
                best_evidence = sentences[max_idx] # The specific sentence that matched
                best_chunk_index = i

        return {
            "score": best_score,
            "evidence": best_evidence,
            "chunk_index": best_chunk_index
        }

    def run_pipeline(self, rag_output: Dict[str, Any]):
        """
        Orchestrates the Full Flow:
        Split -> Match -> Group
        """
        answer_text = rag_output["answer"]
        chunks = rag_output["context_chunks"]

        # Step 1: Split Answer into Claims
        print(f"Splitting answer into claims...")
        atomic_claims = self.get_atomic_claims(answer_text)

        print(atomic_claims)
        processed_claims = []
        
        # Step 2: Granular Matching
        print(f"Matching {len(atomic_claims)} claims against {len(chunks)} chunks...")
        for item in atomic_claims:
            match_result = self.find_best_match(item.claim, chunks)
            
            processed_claims.append({
                "claim": item.claim,
                "score": match_result["score"],
                "evidence_sentence": match_result["evidence"],
                "source_chunk_index": match_result["chunk_index"],
                "original_sentence": item.original_sentence_reference
            })

        # Step 3: Grouping (Preparation for the LLM Judge)
        # We group by 'source_chunk_index' so we can send 1 chunk + 5 claims to the Judge later
        grouped_checks = {}
        for p in processed_claims:
            idx = p["source_chunk_index"]
            if idx not in grouped_checks:
                grouped_checks[idx] = {"chunk_text": chunks[idx] if idx != -1 else "None", "claims": []}
            grouped_checks[idx]["claims"].append(p)

        return {
            "all_claims_scored": processed_claims, # For the Heatmap
            "grouped_for_judge": grouped_checks    # For the verification step
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    # Load your sample data
    with open("data/output/sample2.json", "r") as f:
        data = json.load(f)
    
    # Initialize Detector (Replace with your actual key)
    api_key = os.getenv("OPENAI_API_KEY") 
    detector = HallucinationDetector(api_key)

    # Run on the first example
    result = detector.run_pipeline(data[0])
    # print(result)
    
    # Print results
    print("\n--- RESULTS ---")
    for claim in result["all_claims_scored"]:
        status = "✅" if claim['score'] > 0.50 else "⚠️"
        print(f"{status} [{claim['score']:.2f}] {claim['claim']}")
        print(f"   Evidence: {claim['evidence_sentence'][:50]}...")