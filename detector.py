import os
import json
import nltk
import numpy as np
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal
import ssl

# --- 1. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. SSL Context Fix (MacOS) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

load_dotenv()

# --- 3. NLTK Download Handling ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    try:
        nltk.download('punkt_tab')
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        # We assume basic split might work or fail later, but we log the critical error.

# --- Data Models ---
class AtomicClaim(BaseModel):
    claim: str = Field(..., description="A single, standalone factual statement.")
    original_sentence_reference: str = Field(..., description="The original sentence this claim came from.")

class DecompositionResponse(BaseModel):
    claims: List[AtomicClaim]

class Verdict(BaseModel):
    claim_text: str
    is_supported: Literal["Supported", "Contradicted", "Not Mentioned", "Ambiguous"]
    explanation: str = Field(..., description="A brief explanation based ONLY on the context.")
    correction: str = Field(..., description="If contradicted, provide the correct info. If unsupported, state 'N/A'.")
    confidence_score: float = Field(..., description="Score 0.0-1.0 indicating evidence strength.")

class BatchVerification(BaseModel):
    verdicts: List[Verdict]

# --- Custom Exception ---
class HallucinationException(Exception):
    """Custom exception for pipeline errors."""
    pass

class HallucinationDetector:
    def __init__(self, openai_api_key: Optional[str]):
        if not openai_api_key:
            logger.critical("OpenAI API Key is missing!")
            raise HallucinationException("OpenAI API Key not found. Please set OPENAI_API_KEY env variable.")
            
        try:
            self.client = OpenAI(api_key=openai_api_key)
        except Exception as e:
            raise HallucinationException(f"Failed to initialize OpenAI Client: {e}")

        logger.info("Loading embedding model...")
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load embedding model: {e}")
            raise HallucinationException("Could not load local embedding model. Check internet connection or disk space.")

    def get_atomic_claims(self, text: str) -> List[AtomicClaim]:
        """Uses OpenAI Structured Outputs to break text into facts."""
        if not text or not text.strip():
            logger.warning("Empty text provided for decomposition.")
            return []

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise fact-checker."},
                    {"role": "user", "content": f"Break this text into atomic claims that can be fact checked:\n\n{text}"}
                ],
                response_format=DecompositionResponse,
            )
            parsed = completion.choices[0].message.parsed
            if not parsed or not parsed.claims:
                logger.warning("Model returned no claims.")
                return []
            return parsed.claims

        except (APIConnectionError, RateLimitError) as e:
            logger.error(f"OpenAI Network Error during decomposition: {e}")
            raise HallucinationException("Network error while connecting to OpenAI.")
        except ValidationError as e:
            logger.error(f"Pydantic Validation Error: {e}")
            # If the model output format is broken, we might want to return an empty list or retry
            return [] 
        except Exception as e:
            logger.error(f"Unexpected error in get_atomic_claims: {e}")
            raise HallucinationException(f"Decomposition failed: {e}")

    def find_best_match(self, claim_text: str, retrieved_chunks: List[str]):
        """
        Matches a claim against every SENTENCE in every CHUNK.
        """
        if not retrieved_chunks:
            return {"score": 0.0, "evidence": "", "chunk_index": -1}

        try:
            claim_vector = self.encoder.encode([claim_text])
        except Exception as e:
            logger.error(f"Encoding failed for claim '{claim_text}': {e}")
            return {"score": 0.0, "evidence": "Encoding Error", "chunk_index": -1}
        
        best_score = -1.0
        best_evidence = ""
        best_chunk_index = -1

        for i, chunk in enumerate(retrieved_chunks):
            try:
                sentences = nltk.sent_tokenize(chunk)
                if not sentences: continue
                
                chunk_vectors = self.encoder.encode(sentences)
                
                # Check dimensions to avoid shape mismatch errors
                if chunk_vectors.shape[0] == 0: continue

                scores = cosine_similarity(claim_vector, chunk_vectors)[0]
                
                max_idx = np.argmax(scores)
                max_score = scores[max_idx]

                if max_score > best_score:
                    best_score = float(max_score)
                    best_evidence = sentences[max_idx]
                    best_chunk_index = i
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}. Skipping chunk.")
                continue

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
        if not grouped_data:
            logger.info("No grouped claims to verify.")
            return []
        
        logger.info(f"--- 👨‍⚖️ The Judge is Deliberating ({len(grouped_data)} context groups) ---")

        for chunk_idx, data in grouped_data.items():
            chunk_text = data.get("chunk_text", "")
            claims_list = data.get("claims", [])
            
            if not claims_list: continue

            claims_text_block = "\n".join([f"- {c['claim']}" for c in claims_list])
            logger.info(f"Verifying {len(claims_list)} claims against Chunk {chunk_idx}...")

            try:
                completion = self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": (
                            "You are a strict fact-checking judge. "
                            "Categories:\n"
                            "1. Supported: The context explicitly states this.\n"
                            "2. Contradicted: The context explicitly contradicts this.\n"
                            "3. Not Mentioned: No info found.\n"
                            "4. Ambiguous: Context is too vague.\n"
                            "Rely ONLY on the provided context."
                        )},
                        {"role": "user", "content": f"CONTEXT:\n{chunk_text}\n\nCLAIMS:\n{claims_text_block}"}
                    ],
                    response_format=BatchVerification,
                )
                
                parsed_response = completion.choices[0].message.parsed
                if not parsed_response or not parsed_response.verdicts:
                    logger.warning(f"Judge returned empty verdicts for Chunk {chunk_idx}")
                    # If verification fails, mark these claims as "Error" or "Not Verified"
                    for c in claims_list:
                        final_results.append({
                            "claim": c["claim"],
                            "score": c["score"],
                            "status": "Verification Failed", 
                            "reasoning": "LLM returned invalid format.",
                            "correction": "N/A",
                            "evidence_used": "N/A"
                        })
                    continue

                verdicts = parsed_response.verdicts
                
                # Safety check: Ensure length match (though usually robust)
                if len(verdicts) != len(claims_list):
                    logger.warning(f"Mismatch in verdicts count: Got {len(verdicts)}, Expected {len(claims_list)}")
                    # Map as many as possible
                
                for verdict, original_claim_data in zip(verdicts, claims_list):
                    final_results.append({
                        "claim": original_claim_data["claim"],
                        "score": original_claim_data["score"],
                        "status": verdict.is_supported,
                        "confidence": verdict.confidence_score,
                        "reasoning": verdict.explanation,
                        "correction": verdict.correction,
                        "evidence_used": original_claim_data["evidence_sentence"]
                    })

            except Exception as e:
                logger.error(f"Error during verification step for Chunk {chunk_idx}: {e}")
                # Append failed results so we don't silently lose them
                for c in claims_list:
                    final_results.append({
                        "claim": c["claim"],
                        "score": c["score"],
                        "status": "Verification Error",
                        "reasoning": str(e),
                        "correction": "N/A",
                        "evidence_used": "N/A"
                    })

        return final_results

    def run_pipeline(self, rag_output: Dict[str, Any], threshold: float = 0.4):
        """
        Orchestrates the Full Flow with top-level error catching.
        """
        # 1. Validate Input
        if not rag_output or "answer" not in rag_output or "context_chunks" not in rag_output:
            raise HallucinationException("Invalid Input: rag_output must contain 'answer' and 'context_chunks'")

        answer_text = rag_output["answer"]
        chunks = rag_output["context_chunks"]

        if not chunks:
            logger.warning("No context chunks provided. Cannot verify.")
            return {"verified_results": [], "skipped_low_relevance": []}

        # Step 1: Split
        logger.info(f"Splitting answer into claims...")
        atomic_claims = self.get_atomic_claims(answer_text)
        
        if not atomic_claims:
            logger.warning("No claims found in answer.")
            return {"verified_results": [], "skipped_low_relevance": []}

        processed_claims = []
        
        # Step 2: Granular Matching
        logger.info(f"Matching {len(atomic_claims)} claims against {len(chunks)} chunks...")
        for item in atomic_claims:
            match_result = self.find_best_match(item.claim, chunks)
            
            processed_claims.append({
                "claim": item.claim,
                "score": match_result["score"],
                "evidence_sentence": match_result["evidence"],
                "source_chunk_index": match_result["chunk_index"],
                "original_sentence": item.original_sentence_reference
            })

        # Step 3: Grouping
        grouped_checks = {}
        skipped_claims = []
        
        for p in processed_claims:
            # Filter low scores
            if p["score"] < threshold:
                skipped_claims.append(p)
                continue

            idx = p["source_chunk_index"]
            if idx == -1: 
                skipped_claims.append(p)
                continue

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
    try:
        # Load data safely
        data_path = "data/output/toughCause.json"
        if not os.path.exists(data_path):
            logger.error(f"File not found: {data_path}")
            exit(1)

        with open(data_path, "r") as f:
            data = json.load(f)
        
        api_key = os.getenv("OPENAI_API_KEY") 
        detector = HallucinationDetector(api_key)

        # Run safely
        if isinstance(data, list) and len(data) > 0:
            result = detector.run_pipeline(data[0], threshold=0.45) 
            
            print("\n\n=== 🏁 FINAL REPORT 🏁 ===")
            for res in result["verified_results"]:
                status = res.get("status", "Unknown")
                icon = "✅"
                if status == "Contradicted": icon = "❌"
                elif status == "Ambiguous": icon = "⚠️"
                elif status == "Not Mentioned": icon = "❓"
                elif status == "Verification Error": icon = "🚫"
                
                print(f"\n{icon} [{status}] {res['claim']}")
                print(f"   Reasoning: {res.get('reasoning', 'N/A')}")
        else:
            logger.error("Data file format incorrect. Expected a non-empty list.")

    except HallucinationException as he:
        logger.critical(f"Pipeline Critical Error: {he}")
    except Exception as e:
        logger.critical(f"Unexpected Crash: {e}")