Propositional RAG Verifier: Hallucination Detection Pipeline

Project Overview

    This project implements a robust Hallucination Detection System for 
    Retrieval-Augmented Generation (RAG) applications. Unlike standard evaluation 
    tools that rely on simple similarity scores, this system decomposes answers 
    into Atomic Claims and verifies each claim against retrieved context using 
    a "Sliding Window" alignment strategy and an LLM-based Reasoning Judge.


🛠️ Installation & Setup

Environment Set-up

Use Python 3.10 or higher
Use an OpenAI API Key (required for the Decomposition and Judge modules)

1. Clone the Repository

2. Set up a Virtual Environment
    It is strictly recommended to use a virtual environment to manage dependencies.


3. Install Dependencies

    Install the required packages using the requirements file.
    pip install -r requirements.txt


4. Configure Environment Variables

    Create a .env file in the root directory to store your API credentials.
    touch .env

    Open .env and add your OpenAI API key:

    OPENAI_API_KEY=sk-proj-your-key-here...

Key Features

Atomic Decomposition: 
    Breaks complex sentences into standalone facts to handle "mixed truth" 
    statements.

Sliding Window Alignment: 
    Aligns claims to specific evidence sentences using all-MiniLM-L6-v2 
    embeddings.

Cost-Effective Filtering: 
    Implements a Similarity Gate (tau=0.45) to discard irrelevant claims 
    before expensive LLM verification.

Static Audit Interface: 
    A Streamlit-based UI that visualizes contradictions and ambiguities 
    for end-users.


🛠️ Installation & Setup


Environment Set-up

Use Python 3.10 or higher
Use an OpenAI API Key (required for the Decomposition and Judge modules)

1. Clone the Repository

2. Set up a Virtual Environment

    It is strictly recommended to use a virtual environment to manage dependencies.


3. Install Dependencies

    Install the required packages using the requirements file.
    pip install -r requirements.txt


4. Configure Environment Variables

    Create a .env file in the root directory to store your API credentials.
    touch .env

    Open .env and add your OpenAI API key:

    OPENAI_API_KEY=sk-proj-your-key-here...



Codebase Walkthrough: detector.py

    This document provides a detailed technical breakdown of the 
    HallucinationDetector class and its supporting components. The system is
    designed to verify RAG (Retrieval-Augmented Generation) responses by 
    decomposing them into atomic claims and checking them against retrieved 
    context.

1. Setup and Initialization

    Global Configuration - Before the class definition, we handle environment
    robustness:

    Logging: We replace standard print statements with Python's logging module.
    This provides timestamps and severity levels (INFO, WARNING, ERROR), which 
    is critical for production debugging.

    SSL Context Patch: A specific fix for MacOS environments where Python often
    struggles to verify SSL certificates when downloading NLTK data.

    NLTK Data: We explicitly check for and download the punkt_tab tokenizer.
    This model is required for the "Sliding Window" logic, as it intelligently 
    splits paragraphs into sentences.

    Custom Exceptions
        class HallucinationException(Exception):
            pass

    We define a custom exception to handle pipeline-specific failures 
    (e.g., API keys missing, model loading failures) separately from 
    generic Python errors.

2. Data Models (Pydantic)

    We use Pydantic to define strict schemas for the LLM outputs. This 
    leverages OpenAI's "Structured Outputs" feature to guarantee valid JSON.

        AtomicClaim
            Represents a single, isolated fact extracted from the main answer.

            claim: The verified statement (e.g., "The bridge opened in 1883").
            original_sentence_reference: Used for UI highlighting to map the claim back to the original text.


        Verdict

            The decision structure returned by the LLM Judge.

            is_supported: A Literal enum (Supported, Contradicted, Not Mentioned, Ambiguous) ensuring strict classification.
            correction: If the claim is false, the model must provide the truth from the text.
            confidence_score: A float (0.0 - 1.0) allowing the model to express uncertainty for inferred facts.

3. The HallucinationDetector Class

    This is the core engine of the system.

    __init__(self, openai_api_key)
    Purpose: Bootstraps the heavy models.

    OpenAI Client: Initializes the connection to GPT-4o.

    SentenceTransformer: 
        Loads all-MiniLM-L6-v2. This is a local embedding 
        model. We load it here (once) so we don't reload the 80MB weights for 
        every single query, ensuring performance.

    get_atomic_claims(self, text)
    Stage 1: Decomposition

        Goal: 
            Break complex sentences ("John died of tetanus in 1869") into atomic 
            facts ("John died of tetanus", "This happened in 1869").

        Mechanism: 
            Sends the text to gpt-4o-mini with a specific system prompt 
            instructing it to resolve pronouns (change "He" to "John").

        Why Mini? This is a simple parsing task; gpt-4o-mini is faster and
        cheaper than the large model.


    find_best_match(self, claim_text, retrieved_chunks)
    Stage 2: Alignment (The Sliding Window)

        This is the most mathematically complex function. It solves the 
        "Needle in a Haystack" problem.

        Vectorization: 
            It converts the claim_text into a vector embedding.

        Granularity Loop: 
            It iterates through every context chunk. Crucially, 
            it splits each chunk into individual sentences.

        Cosine Search: 
            It compares the Claim Vector against every Sentence Vector.

        Selection: 
            It identifies the single sentence with the highest similarity score.

        Why? Comparing a short claim to a long paragraph washes out the signal. 
        Comparing Claim-to-Sentence provides high-precision matching.


    verify_claims(self, grouped_data)
    Stage 3: Filtering (The Similarity Gate):

        It checks the score from the alignment step.

        Logic: 
            If score < 0.45, the claim is deemed irrelevant/hallucinated 
            (no grounding in text). It is moved to skipped_claims.

        Benefit: 
            Prevents wasting money sending irrelevant noise to GPT-4o.

        Call Verification: 
            Sends only the surviving claims to the Judge.

        Return: 
            A dictionary separating verified_results (Green/Red/Yellow) 
            from skipped_low_relevance (Orange).

    Stage 4: Verification (The Judge)
        This executes the Batch Verification strategy.

        Grouping: 
            It receives claims grouped by their source chunk (from the 
            orchestration step).

        Prompt Construction: 
            It builds a prompt containing one context chunk and a list 
            of multiple claims.

        Judgment: 
            It calls gpt-4o (the smart model) to reason about the text. 
            It uses the Verdict schema to enforce structured analysis.

        Efficiency: 
            By validating multiple claims in one API call, we significantly 
            reduce latency and token costs.

    
    run_pipeline(self, rag_output, threshold=0.45)
    The Orchestrator
        This function ties all previous steps together:

        Input Validation: 
            Ensures the JSON structure is correct.

        Call Decomposition: 
            Get the atomic claims.

        Call Alignment: 
            Find the best evidence for each claim.


    4. Execution Flow (__main__)
        The script includes a robust entry point for testing:

        Safety Checks: 
            Verifies the data file exists before trying to open it.

        Execution: 
            Runs the pipeline on the given json.

        Reporting: 
            Prints a formatted "Traffic Light" report to the console, using 
            icons (✅, ❌, ⚠️) to visually represent the JSON output.


Codebase Walkthrough: app.py

    This document provides a technical breakdown of the Streamlit frontend. 
    This application serves as the "Static Audit Interface," transforming 
    the raw JSON output from the backend into a human-readable, color-coded 
    report.

    1. Initialization and Caching

        Page Configuration

        st.set_page_config(layout="wide", ...)
        We set the layout to "wide" to maximize horizontal screen real estate,
        allowing the Metrics Row to display the distinct statistics side-by-side
        without cramping.

    The Caching Strategy (@st.cache_resource)

        @st.cache_resource
        def get_detector():
            ...
            return HallucinationDetector(api_key)


        Why this is critical:
        The HallucinationDetector class loads a heavy machine 
        learning model (all-MiniLM-L6-v2, ~80MB) and initializes 
        the OpenAI client.

        Without Caching: Streamlit re-runs the entire script from top to bottom
        on every user interaction (e.g., typing a letter, clicking a button). 
        This would reload the AI model 100 times, crashing the app or making it 
        unbearably slow.

        With @st.cache_resource: Streamlit loads the model once when the app
        starts and keeps it in memory. Subsequent interactions reuse the 
        existing instance instantly.


    2. Input Handling (Sidebar vs. Main)

        We use a "Split View" design pattern to separate Source Data from 
        Analysis Targets.

        Sidebar (Context): Used for the "Truth Source" (the retrieved chunks). 
        We use st.text_area expecting a JSON string.

        Validation: We rely on json.loads() inside the execution block to ensure 
        this text is valid JSON before processing.

        Main Area (Answer): The text being audited (the RAG response). This 
        takes center stage as it is the primary object of analysis.

    3. The Execution Logic

        When the "🔍 Analyze" button is clicked:

        Input Validation: We check if not chunks_input or not answer_input to
        prevent sending empty payloads to the backend.

        Payload Construction: We package the inputs into the dictionary format 
        expected by detector.run_pipeline().

        Pipeline Execution:
        results = detector.run_pipeline(rag_payload, threshold=0.45)


    We call the backend logic. Note the threshold of 0.45, which aligns with our methodology for filtering noise.
    4. The "Static Audit" Visualization

        This section implements the "Traffic Light" design philosophy to reduce cognitive load.
        A. The Metrics Dashboard

            We calculate high-level statistics before showing details.

            col1.metric("Total Claims", num_claims)
            col2.metric("✅ Verified", supported)
            ...
            col5.metric("⛔ Unsupported", total_unsupported, help="...")

            Goal: Give the user an immediate "Health Check" of the text.

            The "Unsupported" Metric: This aggregates two failure modes:

            Claims filtered by the Similarity Gate (Score < 0.45).

            Claims the Judge labeled "Not Mentioned."
            This combined metric represents the total volume of information 
            that has no basis in the provided context.

        B. Priority Rendering (Bad News First)

        The UI is ordered by severity, not by the order of sentences in the 
        text. This is a deliberate "Security Audit" pattern.

        ❌ Contradictions (Red): 
            Rendered using st.error. These are critical failures where the model 
            lied.
        
        ⛔ High Risk / Skipped (Orange): 
            Rendered using st.warning. These are claims that failed the 
            similarity gate (Score < 0.45). We explicitly separate these 
            to show why they failed (Similarity Score vs. Reasoning).

        ⚠️ Ambiguities (Yellow): 
        Rendered using st.warning. These represent logical leaps or inferences.

        ❓ Not Mentioned (Blue): 
            Rendered using st.info. The Judge couldn't find proof, but it passed 
            the similarity gate.

        C. The "Collapsed" Verification

            with st.expander(f"✅ View Verified Claims ({supported})"):

            Design Choice: We hide the "Supported" claims inside a collapsible
            expander.

            Why? In an audit workflow, success is boring; failure is 
            interesting. Hiding the green checks reduces visual noise, 
            forcing the user to focus strictly on the red/yellow errors.

    5. Robustness & Error Handling

        The entire execution block is wrapped in a try...except structure:

        json.JSONDecodeError: Catches invalid JSON in the sidebar (a common 
        user error).

        Exception: Catches backend failures (API timeouts, model errors) and 
        displays them gracefully using st.error instead of crashing the web server.




🚀 Usage Guide

A. Interactive Web App (Recommended)

    Launch the "Static Audit" interface to visualize hallucinations in real-time.

    Run the Streamlit app:
    streamlit run app.py

    Open your browser to the local host port recommended


Workflow:
    Context: Paste your retrieved JSON chunks into the sidebar.

        Example: 
        [
            "The Louvre’s storied past began in the late 12th century when King Philip II Augustus constructed it as a medieval fortress to protect Paris from Anglo-Norman threats. The fortress was a massive structure with a keep and a surrounding wall, intended purely for defensive purposes. However, as the city of Paris grew beyond these walls, the fortress lost its strategic defensive value. In the mid-14th century, Charles V converted the building into a residence, but it was Francis I in 1546 who truly began the transformation of the Louvre into a lavish royal residence in the French Renaissance style. He ordered the demolition of the original medieval keep and commissioned architect Pierre Lescot to build the modern wings. This marked the beginning of the Louvre's status as a primary seat of French royalty.",
            "The final transformation occurred during the tumultuous years of the French Revolution. Following the move of Louis XIV's court to Versailles in 1682, the Louvre had fallen into partial disrepair and was used largely by intellectuals and artists. In 1791, the National Assembly declared that the Louvre should be a national palace dedicated to the preservation of the sciences and the arts. Consequently, the Museum Central des Arts opened on August 10, 1793, displaying 537 paintings and 184 objects of art confiscated from the church and royal family. This established the Louvre as a public museum."
        ]


    Answer: Paste the RAG-generated response into the main text area.
        
        Example:
            "King Philip II Augustus originally built the Louvre in the late 12th century as a defensive fortress. It was later converted into a royal residence by Francis I in 1546. Finally, during the French Revolution, the National Assembly decreed it should become a sanctuary for the arts, and it opened as a museum in 1793."


    Analyze: Click the button to generate the audit report.


B. CLI Analysis (Automated Testing)

    Run the pipeline programmatically for processing.

    We've provided some of the test files that we used to test for edge case
    catches if you would like to try them out.

    On line 318 there is a line
        data_path = "data/output/louvreGood.json"

        Just change the filename if you would like to see other examples.
        WOULD STAY AWAY FROM THE HALLUCINATION_DATASET.JSON
            That was our original super long test file and it is quite expensive to run
            as well as we got rid of the implementation to run that format.

    Execute the script:
        python detector.py
        python3 detector.py

    The script will output a structured log of every verified claim to the 
    terminal.

Reproducing Experiments

    We have provided the detector.py script and some of the dataset to allow for the 
    reproduction of our tests and findings.

    Experimental Setup: The detector.py script is pre-configured with the thresholds 
    ($\tau=0.45$) and prompts used in our report.


🔧 Troubleshooting

Common Issues

    1. HallucinationException: OpenAI API Key not found
        Cause: The .env file is missing or not loading.
        Fix: Ensure .env exists in the root directory and contains 
        OPENAI_API_KEY=....

    2. HallucinationException: Could not load local embedding model
        Cause: Connection issues preventing the download of all-MiniLM-L6-v2.
        Fix: Ensure you have an active internet connection and sufficient disk 
        space (~100MB) for the initial download.

    3. SSL Certificate Errors (Mac OS)
        Cause: Python on Mac sometimes lacks default SSL certificates.
        Fix: The code includes an automatic patch, but if issues persist, run 
        the Install Certificates.command found in your Python application 
        folder.

    4. "Judge returned empty verdicts"
        Cause: The input text may be empty or malformed.
        Fix: Ensure your input JSON contains valid strings for context_chunks 
        and answer.