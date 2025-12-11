Output from rag:
    Outputs a JSON file

Output filtering/set up:
    Sentence splitting into atomic claims:
        Takes the rag output and splits it into claims:
            Run this through a simple llm model that can do this 

    Context alignment: How do we know where the context came from?
        Break all of the context down into json or something:
            Save the full context, then the break down of the sliding window chunks

        Take the claim and run it with cosine similarity against the chunks
            Split the chunks by sentences with a sliding window:
                Move the window and calculate
                Save the context that saved the 