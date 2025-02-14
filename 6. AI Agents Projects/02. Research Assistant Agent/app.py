import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Initialize LLM and search tools
llm = ChatGroq(model="gemma2-9b-it", temperature=0)
tavily_search = TavilySearchResults(max_results=3)

# Pydantic models
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations."
    )

# State initialization
if 'analysts' not in st.session_state:
    st.session_state.analysts = None
if 'current_analyst' not in st.session_state:
    st.session_state.current_analyst = None
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []
if 'final_report' not in st.session_state:
    st.session_state.final_report = None

def generate_analysts(topic: str, max_analysts: int = 3) -> List[Analyst]:
    """Generate AI analysts based on the research topic"""
    analyst_instructions = f"""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
    1. First, review the research topic: {topic}
    2. Determine the most interesting themes.
    3. Pick the top {max_analysts} themes.
    4. Assign one analyst to each theme."""
    
    structured_llm = llm.with_structured_output(Perspectives)
    analysts = structured_llm.invoke(
        [SystemMessage(content=analyst_instructions),
         HumanMessage(content="Generate the set of analysts.")]
    )
    return analysts.analysts

def main():
    st.title("AI Research Assistant")
    st.write("An intelligent system for conducting research with AI analysts")
    
    # Research Topic Input
    with st.form("research_topic"):
        topic = st.text_input("Enter your research topic:", 
                             placeholder="e.g., The impact of AI on healthcare")
        num_analysts = st.slider("Number of analysts:", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Generate Analysts")
        
        if submitted and topic:
            with st.spinner("Generating analysts..."):
                st.session_state.analysts = generate_analysts(topic, num_analysts)
                st.session_state.interview_history = []
                st.session_state.final_report = None
    
    # Display Analysts
    if st.session_state.analysts:
        st.subheader("Your Research Team")
        
        for i, analyst in enumerate(st.session_state.analysts):
            with st.expander(f"Analyst {i+1}: {analyst.name}"):
                st.write(f"**Role:** {analyst.role}")
                st.write(f"**Affiliation:** {analyst.affiliation}")
                st.write(f"**Focus Area:** {analyst.description}")
    
        # Interview Interface
        st.subheader("Interview Your Analysts")
        
        # Analyst Selection
        analyst_names = [analyst.name for analyst in st.session_state.analysts]
        selected_analyst = st.selectbox("Select an analyst to interview:", analyst_names)
        
        # Find the selected analyst object
        st.session_state.current_analyst = next(
            (a for a in st.session_state.analysts if a.name == selected_analyst), 
            None
        )
        
        # Chat Interface
        if st.session_state.current_analyst:
            with st.form("interview_form"):
                user_input = st.text_input("Your question:", 
                                         placeholder="Ask your question...")
                send_message = st.form_submit_button("Send")
                
                if send_message and user_input:
                    # Add user message to history
                    st.session_state.interview_history.append(
                        {"role": "user", "content": user_input}
                    )
                    
                    # Generate response using the research assistant logic
                    with st.spinner("Analyzing and responding..."):
                        # Here you would integrate the full interview logic
                        # For now, we'll use a simple response
                        response = llm.invoke([
                            SystemMessage(content=f"You are {st.session_state.current_analyst.name}, {st.session_state.current_analyst.role}. Respond to questions based on your expertise and perspective."),
                            HumanMessage(content=user_input)
                        ])
                        
                        st.session_state.interview_history.append(
                            {"role": "assistant", "content": response.content}
                        )
            
            # Display Interview History
            st.subheader("Interview History")
            for message in st.session_state.interview_history:
                if message["role"] == "user":
                    st.write("You:", message["content"])
                else:
                    st.write(f"{selected_analyst}:", message["content"])
        
        # Generate Final Report
        if st.button("Generate Final Report"):
            with st.spinner("Generating comprehensive report..."):
                # Here you would integrate the full report generation logic
                report = llm.invoke([
                    SystemMessage(content="Generate a comprehensive research report based on the interviews conducted."),
                    HumanMessage(content=str(st.session_state.interview_history))
                ])
                st.session_state.final_report = report.content
        
        # Display Final Report
        if st.session_state.final_report:
            st.subheader("Final Research Report")
            st.markdown(st.session_state.final_report)

if __name__ == "__main__":
    main()