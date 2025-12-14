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
from typing import Literal
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

class Verdict(BaseModel):
    claim_text: str
    is_supported: Literal["Supported", "Contradicted", "Not Mentioned", "Ambiguous"]
    explanation: str = Field(..., description="A brief explanation of why the claim is supported or not based *only* on the context.")
    correction: str = Field(..., description="If contradicted, provide the correct information from the context. If unsupported, state 'N/A'.")
    confidence_score: float = Field(..., description="A score between 0.0 and 1.0 indicating how strong the evidence is. 1.0 = Explicit Statement, 0.5 = Strong Inference, 0.0 = Guess.")

class BatchVerification(BaseModel):
    verdicts: List[Verdict]


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


    def verify_claims(self, grouped_data: Dict[int, Any]):
        """
        Takes the grouped claims (by chunk) and runs the LLM Judge.
        """
        final_results = []
        
        print(f"\n--- 👨‍⚖️ The Judge is Deliberating ({len(grouped_data)} context groups) ---")

        for chunk_idx, data in grouped_data.items():
            chunk_text = data["chunk_text"]
            claims_list = data["claims"]
            
            # Prepare the prompt inputs
            # We explicitly list the claims we want checked
            claims_text_block = "\n".join([f"- {c['claim']}" for c in claims_list])

            print(f"Verifying {len(claims_list)} claims against Chunk {chunk_idx}...")

            # The Judge Call
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": (
                        "You are a strict fact-checking judge. "
                        "You will receive a Context Text and a list of Claims. "
                        "Categories:\n"
                        "1. Supported: The context explicitly states this or strongly implies it.\n"
                        "2. Contradicted: The context explicitly contradicts this.\n"
                        "3. Not Mentioned: The context does not contain information about this topic.\n"
                        "4. Ambiguous: The context mentions the topic but is too vague or lacks specific details "
                        "to confirm the claim (e.g., text says 'many people' but claim says '500 people')."
                        "Do not use outside knowledge. Rely ONLY on the provided context."
                    )},
                    {"role": "user", "content": f"""
                    CONTEXT:
                    {chunk_text}

                    CLAIMS TO VERIFY:
                    {claims_text_block}
                    """}
                ],
                response_format=BatchVerification,
            )
            
            # Parse results
            verdicts = completion.choices[0].message.parsed.verdicts
            
            # Merge the Judge's verdict back with our original metadata (scores, etc.)
            for verdict, original_claim_data in zip(verdicts, claims_list):
                final_results.append({
                    "claim": original_claim_data["claim"],
                    "score": original_claim_data["score"], # Keep the cosine score for reference
                    "status": verdict.is_supported,
                    "reasoning": verdict.explanation,
                    "correction": verdict.correction,
                    "evidence_used": original_claim_data["evidence_sentence"]
                })

        return final_results

    def run_pipeline(self, rag_output: Dict[str, Any], threshold: float = 0.4):
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
        grouped_checks = {}
        skipped_claims = []
        
        for p in processed_claims:
            # THE FILTER: Only send "relevant" claims to the expensive LLM
            if p["score"] < threshold:
                skipped_claims.append(p)
                continue

            idx = p["source_chunk_index"]
            if idx == -1: continue

            if idx not in grouped_checks:
                grouped_checks[idx] = {"chunk_text": chunks[idx], "claims": []}
            grouped_checks[idx]["claims"].append(p)

        # Step 4: The Judge
        verified_results = self.verify_claims(grouped_checks)
        
        return {
            "verified_results": verified_results,
            "skipped_low_relevance": skipped_claims
        }

# --- Test Block ---
if __name__ == "__main__":
    # Load your sample data
    with open("data/output/finalBoss.json", "r") as f:
        data = json.load(f)
    
    # Initialize Detector (Replace with your actual key)
    api_key = os.getenv("OPENAI_API_KEY") 
    detector = HallucinationDetector(api_key)

    result = detector.run_pipeline(data[0], threshold=0.4) 

    print("\n\n=== 🏁 FINAL REPORT 🏁 ===")
    
    # 1. Show the Verified Claims
    for res in result["verified_results"]:
        icon = "✅"
        if res["status"] == "Contradicted": icon = "❌"
        elif res["status"] == "Ambiguous": icon = "⚠️"
        elif res["status"] == "Not Mentioned": icon = "❓"
        
        print(f"\n{icon} [{res['status']}] {res['claim']}")
        print(f"   Score: {res['score']:.2f}")
        print(f"   Reasoning: {res['reasoning']}")
        if res["status"] == "Contradicted":
            print(f"   Correction: {res['correction']}")

    # 2. Show what we skipped (Efficiency check)
    if result["skipped_low_relevance"]:
        print(f"\n--- Skipped {len(result['skipped_low_relevance'])} claims due to low relevance (< 0.45) ---")
        for s in result["skipped_low_relevance"]:
            print(f"   ⚠️ [{s['score']:.2f}] {s['claim']}")