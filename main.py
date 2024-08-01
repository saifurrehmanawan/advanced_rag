import streamlit as st
from AIAgent_book import AIAgent_book

@st.cache(allow_output_mutation=True)
def load_agent():
    return AIAgent_book()

# Initialize the AI agent
agent_book = load_agent()

def main():
    st.title("AI Agent for Book Queries")
    
    # Input from user
    query = st.text_input("Enter your query:")
    
    if st.button("Get Answer"):
        if query:
            # Get answer and relevant documents
            answer, relevant_docs = agent_book.answer_with_rag(query)
            
            # Display the answer
            st.write("### Answer:")
            st.write(answer)
            
            # Display the relevant context
            st.write("### Relevant Context:")
            for doc in relevant_docs:
                st.write(doc)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
