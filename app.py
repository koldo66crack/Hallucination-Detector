import streamlit as st
import json
import os
from dotenv import load_dotenv
from detector import HallucinationDetector

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="RAG Hallucination Detector",
    page_icon="🕵️‍♀️",
    layout="wide"
)

# --- Initialize Detector ---
@st.cache_resource
def get_detector():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Please set it in your .env file.")
        st.stop()
    return HallucinationDetector(api_key)

detector = get_detector()

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📝 Input Data")
    
    st.subheader("1. Retrieved Context (Chunks)")
    chunks_input = st.text_area(
        "Paste JSON list of strings:", 
        height=300, 
        placeholder='[\n  "The Louvre was built in 1190...",\n  "It is in Paris..."\n]',
        help="Paste a valid JSON list of strings."
    )

    question_input = st.text_input("2. User Question (Optional)", placeholder="When was the Louvre built?")

# --- Main Area ---
st.title("🕵️‍♀️ Hallucination Detector")
st.markdown("Analyze RAG responses and get a static audit report of potential falsehoods.")

st.subheader("3. Generated Answer")
answer_input = st.text_area(
    "Paste the RAG System's Response:", 
    height=150, 
    placeholder="The Louvre was built in 1190 by King Philip..."
)

if st.button("🔍 Analyze for Hallucinations", type="primary"):
    if not chunks_input or not answer_input:
        st.warning("Please provide both Context Chunks and an Answer.")
    else:
        try:
            chunks_list = json.loads(chunks_input)
            if not isinstance(chunks_list, list):
                st.error("Context chunks must be a JSON list of strings.")
                st.stop()
            
            with st.spinner("🤖 Analyzing claims against context..."):
                rag_payload = {
                    "question": question_input,
                    "answer": answer_input,
                    "context_chunks": chunks_list
                }
                
                # RUN PIPELINE
                results = detector.run_pipeline(rag_payload, threshold=0.45)
                
                verified_results = results.get("verified_results", [])
                skipped = results.get("skipped_low_relevance", [])
                
                # --- METRICS ROW ---
                st.divider()
                st.subheader("📊 Analysis Report")
                
                # Calculate Counts
                num_claims = len(verified_results) + len(skipped)
                
                contradictions = sum(1 for r in verified_results if r['status'] == 'Contradicted')
                ambiguities = sum(1 for r in verified_results if r['status'] == 'Ambiguous')
                supported = sum(1 for r in verified_results if r['status'] == 'Supported')
                
                # "Not Mentioned" by Judge
                not_mentioned_judge = sum(1 for r in verified_results if r['status'] == 'Not Mentioned')
                
                # "Low Similarity" (Skipped)
                low_sim_count = len(skipped)
                
                # Combined "Unsupported" Metric
                total_unsupported = not_mentioned_judge + low_sim_count
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Claims", num_claims)
                col2.metric("✅ Verified", supported)
                col3.metric("⚠️ Ambiguous", ambiguities)
                col4.metric("❌ Contradictions", contradictions)
                col5.metric("⛔ Unsupported", total_unsupported, help="Sum of claims 'Not Mentioned' by Judge + claims with Low Similarity found in context.")

                st.divider()

                # --- THE STATIC BREAKDOWN ---
                
                # 1. Show Contradictions (Critical)
                if contradictions > 0:
                    st.subheader("❌ Contradictions Detected")
                    for res in verified_results:
                        if res['status'] == "Contradicted":
                            with st.container():
                                st.error(f"**CLAIM:** {res['claim']}")
                                st.markdown(f"**Correction:** {res.get('correction', 'N/A')}")
                                st.caption(f"**Reasoning:** {res['reasoning']}")
                                st.markdown("---")

                # 2. Show Low Similarity / Unsupported (High Risk Hallucinations)
                # We moved this out of the expander to be visible immediately
                if skipped:
                    st.subheader("⛔ High Risk Hallucinations (No Context Match)")
                    st.markdown("These claims had **zero similarity** (< 0.45) to the retrieved context. They are likely complete fabrications.")
                    for s in skipped:
                        with st.container():
                            # Using a distinct color/box for these
                            st.warning(f"**CLAIM:** {s['claim']}")
                            st.caption(f"**Reasoning:** Similarity Score ({s['score']:.2f}) was too low to be grounded in the context.")
                            st.markdown("---")

                # 3. Show Ambiguities (Warnings)
                if ambiguities > 0:
                    st.subheader("⚠️ Ambiguities (Inference Required)")
                    for res in verified_results:
                        if res['status'] == "Ambiguous":
                            with st.container():
                                st.warning(f"**CLAIM:** {res['claim']}")
                                st.caption(f"**Reasoning:** {res['reasoning']}")
                                st.markdown("---")

                # 4. Show "Not Mentioned" (Judge Verdict)
                # These passed similarity but the Judge couldn't find explicit proof
                not_mentioned_list = [r for r in verified_results if r['status'] == "Not Mentioned"]
                if not_mentioned_list:
                    st.subheader("❓ Not Explicitly Mentioned")
                    for res in not_mentioned_list:
                        with st.container():
                            st.info(f"**CLAIM:** {res['claim']}")
                            st.caption(f"**Reasoning:** {res['reasoning']}")
                            st.markdown("---")
                
                # 5. Show Supported (Collapsed)
                with st.expander(f"✅ View Verified Claims ({supported})"):
                    for res in verified_results:
                        if res['status'] == "Supported":
                            st.markdown(f"**{res['claim']}**")
                            st.caption(f"Reasoning: {res['reasoning']}")
                            st.divider()

        except json.JSONDecodeError:
            st.error("Invalid JSON format in Context Chunks.")
        except Exception as e:
            st.error(f"An error occurred: {e}")