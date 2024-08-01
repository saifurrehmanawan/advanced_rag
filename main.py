import streamlit as st
from AIAgent_book import AIAgent_book

# Initialize the AIAgent_book instance only once
if 'aigent_book' not in st.session_state:
    st.session_state.aigent_book = AIAgent_book()

def main():
    st.title("AI Agent for Book Queries")
    
    # Input from user
    query = st.text_input("Enter your query:")
    
    if st.button("Get Answer"):
        if query:
            # Get answer and relevant documents
            answer, relevant_docs = answer_with_rag(query)
            
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
