import os
import shutil
import streamlit as st

# Ensure the tutor directory is in the Python path if running from a different location
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tutor')))

from tutor.document_loader import load_and_split_documents
from tutor.vector_store import create_or_load_vectorstore
from tutor.qa_chain import build_qa_chain # Uncommented
#from tutor.quiz_generator import generate_quiz # Uncommented
from langchain_google_genai import ChatGoogleGenerativeAI # Replaced OpenAI with ChatGoogleGenerativeAI


# Streamlit config
st.set_page_config(page_title="📘 AI Course Tutor", layout="wide")
st.title("📘 AI Tutor: Personalized Course Assistant")

# Inform user about API key requirement
st.info("Please ensure your `GOOGLE_API_KEY` is set as an environment variable.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your course material (ZIP, PDF, DOCX, PPTX, TXT, Images)",
    type=["zip", "pdf", "docx", "pptx", "txt", "jpg", "jpeg", "png"]
)

# Process uploaded file
if uploaded_file:
    # Create 'data' directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", uploaded_file.name)

    # Save file to disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded: {uploaded_file.name}")

    # Load and split content
    with st.spinner("🔍 Processing your material..."):
        chunks = load_and_split_documents(save_path)
        # Create 'embeddings' directory if it doesn't exist
        os.makedirs("embeddings", exist_ok=True)
        vectorstore = create_or_load_vectorstore(chunks)
        qa_chain = build_qa_chain(vectorstore)

    st.success("✅ Material processed and embedded!")

    # Chat UI
    st.subheader("🤖 Ask a question")
    query = st.text_input("What would you like to know from your materials?")

    if query:
        with st.spinner("Generating answer..."):
            # --- START OF EDIT ---
            # The qa_chain.run() method is deprecated or expects a dictionary in newer LangChain versions.
            # Use direct invocation with a dictionary input.
            result = qa_chain({"query": query})
            # The result from RetrievalQA now contains 'result' and 'source_documents'
            answer = result.get("result", "No answer found.")
            source_docs = result.get("source_documents", [])
            # --- END OF EDIT ---

        st.markdown("### 📘 Answer")
        st.write(answer)

        if source_docs:
            st.markdown("### 📚 Sources")
            for i, doc in enumerate(source_docs):
                st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}")
                st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                # st.write(doc.page_content[:200] + "...") # Show a snippet of content

    # Quiz Button
    with st.expander("📝 Generate a quiz on this topic"):
        if st.button("Generate Quiz"):
            if query: # Ensure there's a query to base the quiz on
                with st.spinner("Creating quiz..."):
                    # Retrieve relevant documents based on the last query
                    docs = vectorstore.similarity_search(query, k=3)
                    context = "\n\n".join([d.page_content for d in docs])
                    # Initialize LLM for quiz generation
                    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0) # Using gemini-pro for quiz
                    quiz = generate_quiz(llm, context, topic=query)
                st.markdown("### 🧠 Quiz")
                st.write(quiz)
            else:
                st.warning("Please ask a question first to generate a quiz on a topic.")

    # Optional cleanup for extracted zip files
    if os.path.exists("temp_extracted"):
        st.info("Cleaning up temporary extracted files...")
        shutil.rmtree("temp_extracted")
        st.success("Temporary files cleaned.")
