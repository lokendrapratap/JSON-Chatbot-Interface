import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from config import Config

# Initialize the GROQ LLM
def initialize_llm():
    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="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",
    )

# Load JSON content
def load_json(file):
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return None

# Create prompt for querying
def create_prompt():
    prompt_template = """
    You are an AI assistant with expertise in answering questions based on structured JSON data.

    Context: {context}

    Data:
    {pairs}

    Question: {query}

    Answer the question based on the content of the JSON data. If the answer is not in the data, respond with "The answer is not available in the provided JSON."
    """
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "pairs", "query"],
    )

# Initialize LLM Chain
def initialize_llm_chain(llm, prompt_template):
    return LLMChain(llm=llm, prompt=prompt_template, verbose=True)

# Query the JSON data
def query_json(llm_chain, json_data, query):
    context = json_data.get("context", "")
    pairs = json.dumps(json_data.get("pairs", []), indent=2)
    response = llm_chain.invoke({
        "context": context,
        "pairs": pairs,
        "query": query,
    })
    return response["text"].strip()
 
# Streamlit UI
def main():
    st.title("JSON Chatbot Interface")
    st.write("Upload a JSON file and ask questions based on its content.")

    # File uploader
    uploaded_file = st.file_uploader("Upload JSON File", type="json")
    if uploaded_file:
        json_data = load_json(uploaded_file)
        if json_data:
            st.success("JSON loaded successfully!")

            # Initialize LLM and chain
            llm = initialize_llm()
            prompt_template = create_prompt()
            llm_chain = initialize_llm_chain(llm, prompt_template)

            # Interactive Q&A
            st.write("\n### Ask questions about the uploaded JSON data")
            query = st.text_input("Your question:")
            if query:
                answer = query_json(llm_chain, json_data, query)
                st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
