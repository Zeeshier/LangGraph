import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage
from typing import TypedDict, List

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API key is missing. Please set it in your .env file.")
    st.stop()

# Initialize the LLM
model = ChatGroq(model="qwen-2.5-32b", temperature=0)

# Define agent state
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Set up memory storage
memory = SqliteSaver.from_conn_string(":memory:")

# Prompts
PLAN_PROMPT = """You are an expert writer tasked with writing a high-level outline of an essay.\nWrite an outline for the given topic. Provide relevant notes or instructions for each section."""

DRAFT_PROMPT = """Using the provided outline, write a full essay draft. Ensure clear structure and coherence."""

CRITIQUE_PROMPT = """Review the draft and provide constructive feedback on clarity, coherence, and structure."""

REVISION_PROMPT = """Revise the essay based on critique and improve its overall quality."""

# Function to generate responses using LLM
def generate_response(prompt, input_text):
    response = model.invoke(prompt + "\n" + input_text)
    return response.content if response else "Error generating response."

# Streamlit UI
def main():
    st.title("Essay Agent")
    topic = st.text_input("Enter your essay topic:")
    
    if st.button("Generate Essay") and topic:
        with st.spinner("Generating essay..."):
            state = AgentState(
                task=topic,
                plan=generate_response(PLAN_PROMPT, topic),
                draft="",
                critique="",
                content=[],
                revision_number=0,
                max_revisions=3
            )
            
            state['draft'] = generate_response(DRAFT_PROMPT, state['plan'])
            state['critique'] = generate_response(CRITIQUE_PROMPT, state['draft'])
            
            st.subheader("Essay Plan")
            st.write(state['plan'])
            
            st.subheader("Draft")
            st.write(state['draft'])
            
            st.subheader("Critique")
            st.write(state['critique'])
    
if __name__ == "__main__":
    main()
