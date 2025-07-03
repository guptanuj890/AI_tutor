from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate # Import PromptTemplate for more control
import streamlit as st

def build_qa_chain(vectorstore):
    """
    Builds a RetrievalQA chain for question answering using the provided vector store.
    Uses ChatGoogleGenerativeAI as the LLM and includes conversational memory.

    Args:
        vectorstore: The FAISS vector store containing document embeddings.

    Returns:
        RetrievalQA: The configured LangChain RetrievalQA chain.
    """
    st.info("Initializing ChatGoogleGenerativeAI model for QA...")
    # Using gemini-pro for question answering. Temperature can be adjusted for creativity.
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    st.success("ChatGoogleGenerativeAI model initialized for QA.")

    st.info("Setting up conversational memory...")
    # ConversationBufferMemory stores the chat history directly.
    # Explicitly set output_key to 'result' to resolve ValueError when return_source_documents=True
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")
    st.success("Conversational memory set up.")

    st.info("Building RetrievalQA chain...")
    # Define a custom prompt template for the QA chain to guide the LLM's responses.
    # This helps ensure the answers are relevant to the provided context.
    # --- START OF EDIT ---
    # Changed {question} to {query} in the template to align with RetrievalQA's default input key.
    qa_template = """You are an AI assistant for a course. Use the following pieces of context
    to answer the user's question. If you don't know the answer, just say that you don't know,
    don't try to make up an answer.
    Keep the answer concise and directly related to the question.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {query}
    Answer:
    Based on the answer and the context, here are some follow-up questions you might consider asking:
    """

    # The PromptTemplate's input_variables should now reflect 'query' instead of 'question'.
    # RetrievalQA with memory will automatically inject 'chat_history' if the template contains {chat_history}.
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "query"], # Changed "question" to "query" here
        template=qa_template,
    )
    # --- END OF EDIT ---

    # RetrievalQA.from_chain_type creates a chain that takes a question and a vectorstore,
    # retrieves relevant documents, and then uses an LLM to answer the question based on those documents.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" combines all relevant documents into a single prompt.
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True, # Set to True to get the source documents used for the answer.
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, # Pass the custom prompt to the chain.
    )
    st.success("RetrievalQA chain built successfully.")
    return qa_chain
