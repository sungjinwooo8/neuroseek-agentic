from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma

from uuid import uuid4

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import StructuredTool



from typing import Annotated, Literal, Sequence

from typing_extensions import TypedDict

from functools import partial




from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph.message import add_messages

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI



from pydantic import BaseModel, Field



from langgraph.graph import END, StateGraph, START

from langgraph.prebuilt import ToolNode, tools_condition



import streamlit as st



st.set_page_config(
    page_title="AI Blog Search", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 AI Blog Search</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent document search powered by Agentic RAG & LangGraph</p>', unsafe_allow_html=True)



# Initialize session state variables if they don't exist

if 'gemini_api_key' not in st.session_state:

    st.session_state.gemini_api_key = ""

if 'vectorstore' not in st.session_state:

    st.session_state.vectorstore = None



def set_sidebar():

    """Setup sidebar for API keys and configuration."""

    with st.sidebar:

        st.markdown("### 🔐 API Configuration")
        st.markdown("---")
        
        gemini_api_key = st.text_input(
            "**Gemini API Key**", 
            type="password", 
            value=st.session_state.get('gemini_api_key', ''),
            help="Enter your Google Gemini API key to enable AI features"
        )

        if st.button("💾 Save API Key", use_container_width=True):

            if gemini_api_key:

                st.session_state.gemini_api_key = gemini_api_key

                st.success("✅ API key saved successfully!")

                # Reset vectorstore when API key changes

                st.session_state.vectorstore = None

            else:

                st.warning("⚠️ Please enter your Gemini API key")
        
        st.markdown("---")
        st.markdown("### 📚 How to Use")
        st.markdown("""
        1. **Enter your API key** above
        2. **Add a blog URL** to index documents
        3. **Ask questions** about the content
        4. Get intelligent answers powered by AI
        """)



def initialize_components():

    """Initialize components that require API keys"""

    if not st.session_state.gemini_api_key:

        return None, None



    try:

        # Initialize embedding model with API key

        embedding_model = GoogleGenerativeAIEmbeddings(

            model="models/embedding-001",

            google_api_key=st.session_state.gemini_api_key

        )



        # Initialize Chroma vector store (persists to disk)

        # Use a persistent directory for the vectorstore

        persist_directory = "./chroma_db"

        

        # Check if vectorstore already exists in session state

        if st.session_state.vectorstore is None:

            db = Chroma(

                persist_directory=persist_directory,

                embedding_function=embedding_model

            )

            st.session_state.vectorstore = db

        else:

            db = st.session_state.vectorstore



        return embedding_model, db

        

    except Exception as e:

        st.error(f"Initialization error: {str(e)}")

        return None, None



class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]



# Edges

## Check Relevance

def grade_documents(state) -> Literal["generate", "rewrite"]:

    """

    Determines whether the retrieved documents are relevant to the question.



    Args:

        state (messages): The current state



    Returns:

        str: A decision for whether the documents are relevant or not

    """



    print("---CHECK RELEVANCE---")



    # Data model

    class grade(BaseModel):

        """Binary score for relevance check."""



        binary_score: str = Field(description="Relevance score 'yes' or 'no'")



    # LLM

    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, model="gemini-2.0-flash", streaming=True)



    # LLM with tool and validation

    llm_with_tool = model.with_structured_output(grade)



    # Prompt

    prompt = PromptTemplate(

        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 

        Here is the retrieved document: \n\n {context} \n\n

        Here is the user question: {question} \n

        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",

        input_variables=["context", "question"],

    )



    # Chain

    chain = prompt | llm_with_tool



    messages = state["messages"]

    last_message = messages[-1]



    question = messages[0].content

    docs = last_message.content



    scored_result = chain.invoke({"question": question, "context": docs})



    score = scored_result.binary_score



    if score == "yes":

        print("---DECISION: DOCS RELEVANT---")

        return "generate"



    else:

        print("---DECISION: DOCS NOT RELEVANT---")

        print(score)

        return "rewrite"

    

# Nodes

## agent node

def agent(state, tools):

    """

    Invokes the agent model to generate a response based on the current state. Given

    the question, it will decide to retrieve using the retriever tool, or simply end.



    Args:

        state (messages): The current state



    Returns:

        dict: The updated state with the agent response appended to messages

    """

    print("---CALL AGENT---")

    messages = state["messages"]

    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, streaming=True, model="gemini-2.0-flash")

    model = model.bind_tools(tools)

    response = model.invoke(messages)

    

    # We return a list, because this will get added to the existing list

    return {"messages": [response]}



## rewrite node

def rewrite(state):

    """

    Transform the query to produce a better question.



    Args:

        state (messages): The current state



    Returns:

        dict: The updated state with re-phrased question

    """



    print("---TRANSFORM QUERY---")

    messages = state["messages"]

    question = messages[0].content



    msg = [

        HumanMessage(

            content=f""" \n 

                    Look at the input and try to reason about the underlying semantic intent / meaning. \n 

                    Here is the initial question:

                    \n ------- \n

                    {question} 

                    \n ------- \n

                    Formulate an improved question: """,

        )

    ]



    # Grader

    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, model="gemini-2.0-flash", streaming=True)

    response = model.invoke(msg)

    return {"messages": [response]}



## generate node

def generate(state):

    """

    Generate answer



    Args:

        state (messages): The current state



    Returns:

         dict: The updated state with re-phrased question

    """

    print("---GENERATE---")

    messages = state["messages"]

    question = messages[0].content

    last_message = messages[-1]



    docs = last_message.content



    # Initialize a Chat Prompt Template (standard RAG prompt)

    prompt_template = PromptTemplate(

        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:""",

        input_variables=["question", "context"],

    )



    # Initialize a Generator (i.e. Chat Model)

    chat_model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, model="gemini-2.0-flash", temperature=0, streaming=True)



    # Initialize a Output Parser

    output_parser = StrOutputParser()

    

    # RAG Chain

    rag_chain = prompt_template | chat_model | output_parser



    response = rag_chain.invoke({"context": docs, "question": question})

    

    return {"messages": [response]}



# graph function

def get_graph(retriever_tool):

    tools = [retriever_tool]  # Create tools list here

    

    # Define a new graph

    workflow = StateGraph(AgentState)



    # Use partial to pass tools to the agent function

    workflow.add_node("agent", partial(agent, tools=tools))

    

    # Rest of the graph setup remains the same

    retrieve = ToolNode(tools)

    workflow.add_node("retrieve", retrieve)

    workflow.add_node("rewrite", rewrite)  # Re-writing the question

    workflow.add_node(

        "generate", generate

    )  # Generating a response after we know the documents are relevant

    # Call agent node to decide to retrieve or not

    workflow.add_edge(START, "agent")



    # Decide whether to retrieve

    workflow.add_conditional_edges(

        "agent",

        # Assess agent decision

        tools_condition,

        {

            # Translate the condition outputs to nodes in our graph

            "tools": "retrieve",

            END: END,

        },

    )



    # Edges taken after the `action` node is called.

    workflow.add_conditional_edges(

        "retrieve",

        # Assess agent decision

        grade_documents,

    )

    workflow.add_edge("generate", END)

    workflow.add_edge("rewrite", "agent")



    # Compile

    graph = workflow.compile()



    return graph



def generate_message(graph, inputs):

    generated_message = ""



    for output in graph.stream(inputs):

        for key, value in output.items():

            if key == "generate" and isinstance(value, dict):

                messages = value.get("messages", [])
                if messages:
                    # Handle both string and message object responses
                    last_msg = messages[-1]
                    if isinstance(last_msg, str):
                        generated_message = last_msg
                    elif hasattr(last_msg, 'content'):
                        generated_message = last_msg.content
                    else:
                        generated_message = str(last_msg)

    # If no message found in generate node, try to get from final state
    if not generated_message:
        final_state = graph.invoke(inputs)
        if "messages" in final_state and final_state["messages"]:
            last_msg = final_state["messages"][-1]
            if isinstance(last_msg, str):
                generated_message = last_msg
            elif hasattr(last_msg, 'content'):
                generated_message = last_msg.content
            else:
                generated_message = str(last_msg)
    
    return generated_message



def add_documents_to_vectorstore(url, db):

    try:

        docs = WebBaseLoader(url).load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(

            chunk_size=100, chunk_overlap=50

        )

        doc_chunks = text_splitter.split_documents(docs)

        uuids = [str(uuid4()) for _ in range(len(doc_chunks))]

        db.add_documents(documents=doc_chunks, ids=uuids)

        return True

    except Exception as e:

        st.error(f"Error adding documents: {str(e)}")

        return False



def main():

    set_sidebar()



    # Check if API key is set

    if not st.session_state.gemini_api_key:

        st.info("👈 Please configure your Gemini API key in the sidebar to get started")

        return



    # Initialize components

    embedding_model, db = initialize_components()

    if not all([embedding_model, db]):

        return



    # Initialize retriever and tools

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Create retriever tool manually using StructuredTool
    def retrieve_documents(query: str) -> str:
        """Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = StructuredTool.from_function(
        func=retrieve_documents,
        name="retrieve_blog_posts",
        description="Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs."
    )

    tools = [retriever_tool]



    # URL input section
    st.markdown("### 📄 Add Documents")
    st.markdown("Enter a blog URL to index and search through its content")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        url = st.text_input(
            "**Blog URL**",
            placeholder="https://example.com/blog-post",
            label_visibility="collapsed"
        )
    
    with col2:
        add_url = st.button("➕ Add", use_container_width=True)
    
    if add_url:
        if url:
            with st.spinner("🔄 Processing and indexing documents..."):
                if add_documents_to_vectorstore(url, db):
                    st.success("✅ Documents added successfully! You can now ask questions about this content.")
                else:
                    st.error("❌ Failed to add documents. Please check the URL and try again.")
        else:
            st.warning("⚠️ Please enter a valid URL")



    # Query section
    st.markdown("---")
    st.markdown("### 💬 Ask Questions")
    st.markdown("Query the indexed documents using natural language")
    
    graph = get_graph(retriever_tool)

    query = st.text_area(
        "**Your Question**",
        placeholder="What are the main concepts discussed in this blog?",
        label_visibility="collapsed",
        height=120
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        submit_query = st.button("🚀 Search", use_container_width=True)

    if submit_query:
        if not query:
            st.warning("⚠️ Please enter a question")
            return

        inputs = {"messages": [HumanMessage(content=query)]}

        with st.spinner("🤔 Thinking... Analyzing documents and generating response..."):
            try:
                response = generate_message(graph, inputs)
                
                # Display response in a styled box
                st.markdown("---")
                st.markdown("### 💡 Answer")
                st.markdown(f"""
                <div class="info-box">
                    {response}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")






if __name__ == "__main__":

    main()

