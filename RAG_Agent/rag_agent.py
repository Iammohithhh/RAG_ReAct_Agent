import streamlit as st
import os
import tempfile
import logging
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime, timezone
import json
import shutil
import traceback
import uuid
from operator import add
import re
import importlib.util

# Core imports
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from groq import Groq

# Configure logging to suppress warnings
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

class RAGState(TypedDict):
    """State for the RAG ReAct agent with proper annotations."""
    messages: Annotated[List[Dict[str, Any]], add]  # Updated type hint
    query: str
    thought: str
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]
    final_answer: Optional[str]
    iteration_count: int
    max_iterations: int
    tools_used: Annotated[List[str], add]
    context: Dict[str, Any]
    documents: Annotated[List[Document], add]
    retrieved_chunks: Annotated[List[str], add]
    document_metadata: Dict[str, Any]

class DocumentManager:
    """Manages document loading, processing, and retrieval."""
    
    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = self._initialize_embeddings(embeddings_model)
        self.vectorstore = None
        self.retriever = None
        self.persist_directory = None
        self.loaded_documents = []
        
    def _initialize_embeddings(self, model_name: str):
        """Initialize embeddings with comprehensive fallback."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            test_embedding = embeddings.embed_query("test")
            if test_embedding:
                return embeddings
        except Exception as e:
            logger.warning(f"Primary embeddings failed: {e}")
        
        fallback_models = [
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/distilbert-base-nli-mean-tokens"
        ]
        
        for fallback_model in fallback_models:
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=fallback_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                test_embedding = embeddings.embed_query("test")
                if test_embedding:
                    return embeddings
            except Exception as e:
                logger.warning(f"Fallback {fallback_model} failed: {e}")
                continue
        
        return self._create_simple_embeddings()
    
    def _create_simple_embeddings(self):
        """Create simple TF-IDF based embeddings as last resort."""
        if not importlib.util.find_spec("sklearn"):
            logger.warning("scikit-learn not installed, using dummy embeddings")
            return self._create_dummy_embeddings()
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            class SimpleTFIDFEmbeddings:
                def __init__(self):
                    self.vectorizer = TfidfVectorizer(
                        max_features=384, 
                        stop_words='english',
                        ngram_range=(1, 2),
                        max_df=0.95,
                        min_df=2
                    )
                    self.is_fitted = False
                    
                def embed_documents(self, texts):
                    if not texts:
                        return []
                    
                    if not self.is_fitted:
                        self.vectorizer.fit(texts)
                        self.is_fitted = True
                    
                    try:
                        vectors = self.vectorizer.transform(texts).toarray()
                        return vectors.tolist()
                    except:
                        return [[0.0] * 384 for _ in texts]
                    
                def embed_query(self, text):
                    if not self.is_fitted:
                        return [0.0] * 384
                    
                    try:
                        vector = self.vectorizer.transform([text]).toarray()[0]
                        return vector.tolist()
                    except:
                        return [0.0] * 384
            
            return SimpleTFIDFEmbeddings()
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF embeddings: {e}")
            return self._create_dummy_embeddings()
    
    def _create_dummy_embeddings(self):
        """Create dummy embeddings as final fallback."""
        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 384
        
        return DummyEmbeddings()
    
    def load_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Load and process PDF document."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            if not docs:
                raise ValueError("No content found in PDF")
            
            valid_docs = [doc for doc in docs if doc.page_content and len(doc.page_content.strip()) > 10]
            if not valid_docs:
                raise ValueError("No valid content found in PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(valid_docs)
            filtered_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 20]
            
            self.loaded_documents = filtered_docs
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def load_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Load and process text content."""
        try:
            if not text or len(text.strip()) < 10:
                raise ValueError("Text content is too short or empty")
            
            doc = Document(
                page_content=text.strip(), 
                metadata={"source": "text_input", "length": len(text)}
            )
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents([doc])
            filtered_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 20]
            
            if not filtered_docs:
                filtered_docs = [doc]
            
            self.loaded_documents = filtered_docs
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error loading text: {e}")
            raise
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vectorstore from documents."""
        try:
            if not documents:
                raise ValueError("No documents provided for vectorstore creation")
            
            self.cleanup()
            self.persist_directory = tempfile.mkdtemp(prefix="rag_vectorstore_")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="rag_collection"
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return self.vectorstore, self.retriever
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            self.cleanup()
            raise
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if not self.retriever:
            return []
        
        try:
            docs = self.retriever.invoke(query)
            return docs[:k] if docs else []
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.persist_directory and os.path.exists(self.persist_directory):
            try:
                self.vectorstore = None
                self.retriever = None
                shutil.rmtree(self.persist_directory, ignore_errors=True)
                logger.info(f"Cleaned up vectorstore directory")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
        self.persist_directory = None

class RAGReActAgent:
    """RAG-enhanced ReAct agent using LangGraph and Groq."""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192", timezone_str: str = "UTC"):
        if not groq_api_key:
            raise ValueError("Groq API key is required")
        
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1000,
            timeout=30
        )
        self.groq_client = Groq(api_key=groq_api_key)
        self.doc_manager = DocumentManager()
        
        # Set timezone
        self.timezone = timezone.utc
        if importlib.util.find_spec("pytz"):
            try:
                import pytz
                self.timezone = pytz.timezone(timezone_str)
            except pytz.exceptions.UnknownTimeZoneError:
                logger.warning(f"Invalid timezone {timezone_str}, defaulting to UTC")
        
        self.tools = self._setup_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.max_iterations = 7
        self.workflow = self._create_workflow()
        
    def _setup_tools(self) -> List[Tool]:
        """Set up tools for the agent."""
        tools = []
        
        def retrieve_documents(query: str) -> str:
            """Retrieve relevant documents from loaded PDF/text."""
            try:
                if not query or not query.strip():
                    return "Please provide a valid search query."
                
                docs = self.doc_manager.retrieve_documents(query.strip())
                if not docs:
                    return "No relevant documents found. Make sure documents are loaded first."
                
                result = f"Found {len(docs)} relevant document chunks:\n\n"
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    result += f"Chunk {i}:\n{content}\n\n"
                
                return result
            except Exception as e:
                logger.error(f"Document retrieval error: {e}")
                return "Document retrieval is currently unavailable."
        
        tools.append(Tool(
            name="retrieve_documents",
            func=retrieve_documents,
            description="Retrieve relevant document chunks from loaded PDF or text based on a query."
        ))
        
        def summarize_document() -> str:
            """Summarize the loaded document."""
            try:
                if not self.doc_manager.loaded_documents:
                    return "No documents loaded to summarize. Please load a PDF or text first."
                
                # Use up to 5 chunks or 4000 characters
                content = ""
                char_limit = 4000
                chunk_count = min(5, len(self.doc_manager.loaded_documents))
                
                for doc in self.doc_manager.loaded_documents[:chunk_count]:
                    if len(content) + len(doc.page_content) <= char_limit:
                        content += doc.page_content + "\n\n"
                    else:
                        remaining = char_limit - len(content)
                        if remaining > 100:  # Only add if significant space remains
                            content += doc.page_content[:remaining] + "\n\n"
                        break
                
                if not content.strip():
                    return "No valid content found in loaded documents."
                
                prompt = f"""Summarize the following document content concisely, capturing the main points and key information in a structured format (e.g., bullet points or short paragraphs):

{content}

Provide a clear, accurate summary without adding external information."""
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.groq_client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=700,  # Increased for better summaries
                            temperature=0.1
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        logger.error(f"Summary attempt {attempt + 1} failed: {e}")
                        if attempt == max_retries - 1:
                            return f"Error summarizing document: {str(e)}"
                        time.sleep(1)  # Wait before retrying
                
                return "Document summarization failed after multiple attempts."
                
            except Exception as e:
                logger.error(f"Document summarization error: {e}")
                return f"Document summarization failed: {str(e)}"
        
        tools.append(Tool(
            name="summarize_document",
            func=summarize_document,
            description="Provide a summary of the loaded document. Ensure a document is loaded first."
        ))
        
        def safe_web_search(query: str) -> str:
            """Search the web for information."""
            try:
                if not query or not query.strip():
                    return "Please provide a valid search query."
                
                search = DuckDuckGoSearchRun()
                result = search.run(query.strip())
                return result if result else "No web search results found."
            except Exception as e:
                logger.error(f"Web search error: {e}")
                return "Web search is currently unavailable."
        
        tools.append(Tool(
            name="web_search",
            func=safe_web_search,
            description="Search the web for current information about a topic."
        ))
        
        def safe_wikipedia_search(query: str) -> str:
            """Search Wikipedia for information."""
            try:
                if not query or not query.strip():
                    return "Please provide a valid search query."
                
                wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
                result = wikipedia.run(query.strip())
                return result if result else "No Wikipedia results found."
            except Exception as e:
                logger.error(f"Wikipedia search error: {e}")
                return "Wikipedia search is currently unavailable."
        
        tools.append(Tool(
            name="wikipedia_search",
            func=safe_wikipedia_search,
            description="Search Wikipedia for information about a topic."
        ))
        
        def safe_calculator(expression: str) -> str:
            """Perform mathematical calculations."""
            try:
                if not expression or not expression.strip():
                    return "Please provide a valid mathematical expression."
                
                allowed_chars = "0123456789+-*/.() "
                clean_expr = ''.join(c for c in expression if c in allowed_chars)
                
                if not clean_expr:
                    return "Invalid mathematical expression."
                
                result = eval(clean_expr, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except ZeroDivisionError:
                return "Error: Division by zero."
            except Exception as e:
                logger.error(f"Calculator error: {e}")
                return "Error: Invalid mathematical expression."
        
        tools.append(Tool(
            name="calculator",
            func=safe_calculator,
            description="Perform basic arithmetic calculations (addition, subtraction, multiplication, division)."
        ))
        
        def get_current_datetime(format_request: str = "") -> str:
            """Get current date and time with flexible formatting options."""
            try:
                now = datetime.now(self.timezone)
                format_request = format_request.lower().strip()
                
                if "date only" in format_request or "just date" in format_request:
                    return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')})"
                elif "time only" in format_request or "just time" in format_request:
                    return f"Current time: {now.strftime('%H:%M:%S')} ({now.strftime('%I:%M:%S %p')})"
                elif "iso" in format_request or "utc" in format_request:
                    utc_now = datetime.now(timezone.utc)
                    return f"Current UTC time (ISO format): {utc_now.isoformat()}"
                elif "timestamp" in format_request or "unix" in format_request:
                    timestamp = int(now.timestamp())
                    return f"Current Unix timestamp: {timestamp} (corresponds to {now.strftime('%Y-%m-%d %H:%M:%S %Z')})"
                elif "detailed" in format_request or "full" in format_request:
                    return (f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')} "
                           f"({now.strftime('%Y-%m-%d %H:%M:%S %Z')})")
                else:
                    return (f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} "
                           f"({now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')})")
                    
            except Exception as e:
                logger.error(f"DateTime error: {e}")
                now = datetime.now()
                return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        tools.append(Tool(
            name="get_datetime",
            func=get_current_datetime,
            description="Get the current date and time. You can specify format: 'date only', 'time only', 'detailed', 'iso', 'timestamp', or leave empty for default format."
        ))
        
        def world_clock(timezone_query: str = "") -> str:
            """Get time in different timezones around the world."""
            if not importlib.util.find_spec("pytz"):
                return "World clock feature requires pytz library. Showing local time instead: " + get_current_datetime()
            
            try:
                import pytz
                timezone_query = timezone_query.lower().strip()
                
                timezone_map = {
                    "new york": "America/New_York",
                    "ny": "America/New_York",
                    "est": "America/New_York",
                    "london": "Europe/London",
                    "uk": "Europe/London",
                    "gmt": "Europe/London",
                    "tokyo": "Asia/Tokyo",
                    "japan": "Asia/Tokyo",
                    "jst": "Asia/Tokyo",
                    "sydney": "Australia/Sydney",
                    "australia": "Australia/Sydney",
                    "los angeles": "America/Los_Angeles",
                    "la": "America/Los_Angeles",
                    "pst": "America/Los_Angeles",
                    "chicago": "America/Chicago",
                    "cst": "America/Chicago",
                    "denver": "America/Denver",
                    "mst": "America/Denver",
                    "paris": "Europe/Paris",
                    "berlin": "Europe/Berlin",
                    "moscow": "Europe/Moscow",
                    "dubai": "Asia/Dubai",
                    "singapore": "Asia/Singapore",
                    "hong kong": "Asia/Hong_Kong",
                    "mumbai": "Asia/Kolkata",
                    "india": "Asia/Kolkata",
                    "ist": "Asia/Kolkata"
                }
                
                if not timezone_query:
                    cities = [
                        ("New York", "America/New_York"),
                        ("London", "Europe/London"),
                        ("Paris", "Europe/Paris"),
                        ("Tokyo", "Asia/Tokyo"),
                        ("Sydney", "Australia/Sydney"),
                        ("Los Angeles", "America/Los_Angeles")
                    ]
                    
                    result = "Current time in major world cities:\n\n"
                    for city, tz_name in cities:
                        try:
                            tz = pytz.timezone(tz_name)
                            city_time = datetime.now(tz)
                            result += f"{city}: {city_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                        except:
                            continue
                    
                    return result
                
                tz_name = timezone_map.get(timezone_query)
                if not tz_name:
                    try:
                        pytz.timezone(timezone_query)
                        tz_name = timezone_query
                    except:
                        return f"Timezone '{timezone_query}' not recognized. Try: new york, london, tokyo, sydney, los angeles, paris, etc."
                
                tz = pytz.timezone(tz_name)
                local_time = datetime.now(tz)
                
                return f"Current time in {timezone_query.title()}: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({local_time.strftime('%A, %B %d, %Y at %I:%M:%S %p')})"
                
            except Exception as e:
                logger.error(f"World clock error: {e}")
                return "World clock is currently unavailable. " + get_current_datetime()
        
        tools.append(Tool(
            name="world_clock",
            func=world_clock,
            description="Get current time in different timezones. Specify a city/timezone like 'new york', 'london', 'tokyo', or leave empty for major cities."
        ))
        
        return tools
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(RAGState)
        
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("action", self._action_node)
        workflow.add_node("observation", self._observation_node)
        workflow.add_node("final_answer", self._final_answer_node)
        
        workflow.add_edge("action", "observation")
        workflow.add_edge("observation", "reasoning")
        workflow.add_edge("final_answer", END)
        
        workflow.add_conditional_edges(
            "reasoning",
            self._should_continue,
            {
                "continue": "action",
                "end": "final_answer"
            }
        )
        
        workflow.set_entry_point("reasoning")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _reasoning_node(self, state: RAGState) -> RAGState:
        """Process reasoning step."""
        query = state["query"]
        iteration = state["iteration_count"]
        
        context = self._build_context(state)
        
        reasoning_prompt = f"""You are a helpful AI assistant using the ReAct framework to answer: "{query}"

{context}

Available tools:
- retrieve_documents: Get relevant chunks from loaded documents
- summarize_document: Get a summary of the loaded document  
- web_search: Search the web for current information
- wikipedia_search: Search Wikipedia for information
- calculator: Perform mathematical calculations
- get_datetime: Get current date and time (specify format like 'date only', 'time only', 'detailed', 'iso', 'timestamp')
- world_clock: Get time in different timezones (specify city like 'new york', 'london', 'tokyo')

Current iteration: {iteration + 1}/{state['max_iterations']}

Think step by step about what you need to do next. Format your response exactly as:

Thought: [Your reasoning about what to do next]
Action: [The exact tool name, or 'Final Answer' if ready to conclude]
Action Input: [Input for the action, if applicable]"""
        
        try:
            response = self.llm.invoke(reasoning_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            thought, action, action_input = self._parse_response(content)
            
            if not thought:
                thought = f"Processing query about: {query}"
            
            if not action or iteration >= state["max_iterations"] - 1:
                action = "Final Answer"
                action_input = ""
            
            if action != "Final Answer" and action not in self.tool_map:
                action = self._find_matching_tool(action)
                if not action:
                    action = "Final Answer"
                    action_input = ""
            
            if action != "Final Answer" and not action_input:
                action_input = query
            
            new_state = state.copy()
            new_state["thought"] = thought
            new_state["action"] = action
            new_state["action_input"] = action_input
            new_state["iteration_count"] = iteration + 1
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            new_state = state.copy()
            new_state["thought"] = "Encountered an error during reasoning"
            new_state["action"] = "Final Answer"
            new_state["action_input"] = ""
            new_state["iteration_count"] = iteration + 1
            return new_state
    
    def _find_matching_tool(self, action: str) -> Optional[str]:
        """Find matching tool name from available tools."""
        if not action:
            return None
            
        action_lower = action.lower().strip()
        
        if action in self.tool_map:
            return action
        
        for tool_name in self.tool_map.keys():
            if action_lower in tool_name.lower() or tool_name.lower() in action_lower:
                return tool_name
        
        mapping = {
            "document": "retrieve_documents",
            "search_documents": "retrieve_documents",
            "search": "web_search",
            "calculate": "calculator",
            "datetime": "get_datetime",
            "time": "get_datetime",
            "date": "get_datetime",
            "clock": "world_clock",
            "timezone": "world_clock",
            "world time": "world_clock",
            "wiki": "wikipedia_search",
            "wikipedia": "wikipedia_search",
            "summary": "summarize_document",
            "summarize": "summarize_document"
        }
        
        for key, tool_name in mapping.items():
            if key in action_lower:
                return tool_name
        
        return None
    
    def _parse_response(self, content: str):
        """Parse the LLM response for thought, action, and action input."""
        thought = ""
        action = ""
        action_input = ""
        
        # Use regex for more robust parsing
        thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|$)', content, re.DOTALL)
        action_match = re.search(r'Action:\s*(.*?)(?=\nAction Input:|$)', content, re.DOTALL)
        input_match = re.search(r'Action Input:\s*(.*)', content, re.DOTALL)
        
        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip()
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input
    
    def _action_node(self, state: RAGState) -> RAGState:
        """Process action step."""
        return state
    
    def _observation_node(self, state: RAGState) -> RAGState:
        """Execute the action and observe results."""
        action = state["action"]
        action_input = state["action_input"]
        
        if action == "Final Answer":
            return state
        
        observation = "Tool not found or action failed."
        
        try:
            if action in self.tool_map:
                tool = self.tool_map[action]
                try:
                    observation = tool.run(action_input or "")
                    new_state = state.copy()
                    new_state["tools_used"] = state["tools_used"] + [tool.name]
                    new_state["observation"] = observation
                    return new_state
                except Exception as e:
                    logger.error(f"Error using tool {action}: {e}")
                    observation = f"Error using {action}: {str(e)}"
            else:
                observation = f"Tool '{action}' not found. Available tools: {', '.join(self.tool_map.keys())}"
                
        except Exception as e:
            logger.error(f"Error in observation node: {e}")
            observation = f"An error occurred while executing the action: {str(e)}"
        
        new_state = state.copy()
        new_state["observation"] = observation
        return new_state
    
    def _final_answer_node(self, state: RAGState) -> RAGState:
        """Generate the final answer."""
        query = state["query"]
        context = self._build_context(state)
        
        final_prompt = f"""Based on your research for the query: "{query}"

{context}

Please provide a comprehensive, helpful final answer. Use all the information you've gathered to give a complete and accurate response.

Final Answer:"""
        
        try:
            response = self.llm.invoke(final_prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            final_answer = f"I apologize, but I encountered an issue generating the final answer. Based on the available information, I'll provide what I can about: {query}"
        
        new_state = state.copy()
        new_state["final_answer"] = final_answer
        return new_state
    
    def _should_continue(self, state: RAGState) -> str:
        """Determine whether to continue or end."""
        action = state.get("action", "")
        iteration = state["iteration_count"]
        
        if action == "Final Answer" or iteration >= state["max_iterations"]:
            return "end"
        return "continue"
    
    def _build_context(self, state: RAGState) -> str:
        """Build context from previous iterations."""
        context_parts = []
        
        if state.get("thought"):
            context_parts.append(f"Previous thought: {state['thought']}")
        
        if state.get("observation"):
            context_parts.append(f"Previous observation: {state['observation']}")
        
        if state.get("tools_used"):
            unique_tools = list(set(state['tools_used']))
            context_parts.append(f"Tools used so far: {', '.join(unique_tools)}")
        
        return "\n".join(context_parts) if context_parts else "Starting analysis..."
    
    def load_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Load a PDF document."""
        try:
            split_docs = self.doc_manager.load_pdf(pdf_path, chunk_size, chunk_overlap)
            self.doc_manager.create_vectorstore(split_docs)
            return f"Successfully loaded PDF with {len(split_docs)} chunks"
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            self.doc_manager.cleanup()
            return f"Error loading PDF: {str(e)}"
    
    def load_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Load text content."""
        try:
            split_docs = self.doc_manager.load_text(text, chunk_size, chunk_overlap)
            self.doc_manager.create_vectorstore(split_docs)
            return f"Successfully loaded text with {len(split_docs)} chunks"
        except Exception as e:
            logger.error(f"Error loading text: {e}")
            self.doc_manager.cleanup()
            return f"Error loading text: {str(e)}"
    
    def run(self, query: str) -> str:
        """Run the RAG ReAct agent."""
        if not query or not query.strip():
            return "Please provide a valid question."
        
        initial_state = RAGState(
            messages=[],
            query=query.strip(),
            thought="",
            action=None,
            action_input=None,
            observation=None,
            final_answer=None,
            iteration_count=0,
            max_iterations=self.max_iterations,
            tools_used=[],
            context={},
            documents=[],
            retrieved_chunks=[],
            document_metadata={}
        )
        
        try:
            result = self.workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": str(uuid.uuid4())}}
            )
            return result.get("final_answer", "I couldn't generate a proper answer. Please try rephrasing your question.")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"I encountered an issue while processing your request: {str(e)}. Please try again or rephrase your question."
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.doc_manager.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG ReAct Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG ReAct Agent")
    st.markdown("*AI Assistant with Document Analysis and Web Search*")
    st.markdown("---")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'document_info' not in st.session_state:
        st.session_state.document_info = ""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/keys",
            placeholder="gsk_..."
        )
        
        model_name = st.selectbox(
            "Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
            help="Select the model to use"
        )
        
        timezone_options = [
            "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
            "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Moscow",
            "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Asia/Dubai",
            "Australia/Sydney", "Australia/Melbourne"
        ]
        
        selected_timezone = st.selectbox(
            "Timezone",
            timezone_options,
            index=0,
            help="Select your timezone for date/time operations"
        )
        
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
            max_iterations = st.slider("Max Iterations", 3, 10, 5, 1)
        
        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            if groq_api_key:
                try:
                    with st.spinner("Initializing agent..."):
                        st.session_state.agent = RAGReActAgent(
                            groq_api_key=groq_api_key,
                            model_name=model_name,
                            timezone_str=selected_timezone
                        )
                        st.session_state.agent.max_iterations = max_iterations
                    st.success("✅ Agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing agent: {str(e)}")
            else:
                st.error("❌ Please enter your Groq API key")
        
        st.markdown("---")
        
        st.header("📄 Document Loading")
        
        with st.expander("📝 Load Text"):
            text_input = st.text_area(
                "Enter text content",
                height=150,
                placeholder="Paste your text here..."
            )
            
            if st.button("📝 Load Text", use_container_width=True):
                if st.session_state.agent and text_input.strip():
                    with st.spinner("Loading text..."):
                        result = st.session_state.agent.load_text(text_input, chunk_size, chunk_overlap)
                    if "Successfully" in result:
                        st.success(result)
                        st.session_state.documents_loaded = True
                        st.session_state.document_info = f"Text loaded: {len(text_input)} characters"
                    else:
                        st.error(result)
                else:
                    st.error("❌ Please initialize agent and enter text")
        
        with st.expander("📄 Upload PDF"):
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document"
            )
            
            if st.button("📄 Load PDF", use_container_width=True):
                if st.session_state.agent and uploaded_file:
                    with st.spinner("Loading PDF..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            result = st.session_state.agent.load_pdf(tmp_file_path, chunk_size, chunk_overlap)
                            if "Successfully" in result:
                                st.success(result)
                                st.session_state.documents_loaded = True
                                st.session_state.document_info = f"PDF loaded: {uploaded_file.name}"
                            else:
                                st.error(result)
                        except Exception as e:
                            st.error(f"Error loading PDF: {str(e)}")
                        finally:
                            try:
                                os.unlink(tmp_file_path)
                            except Exception as e:
                                logger.warning(f"Failed to delete temporary file: {e}")
                else:
                    st.error("❌ Please initialize agent and upload PDF")
        
        if st.session_state.documents_loaded:
            st.success(f"✅ {st.session_state.document_info}")
        else:
            st.info("ℹ️ No documents loaded")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.cleanup()
            st.rerun()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("💬 Chat Interface")
        
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        
        if query := st.chat_input("Ask me anything..."):
            if st.session_state.agent:
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response = st.session_state.agent.run(query)
                    st.write(response)
                
                st.session_state.chat_history.append((query, response))
            else:
                st.error("❌ Please initialize the agent first")
    
    with col2:
        st.header("ℹ️ Status & Tools")
        
        if st.session_state.agent:
            st.success("🟢 Agent: Online")
            try:
                current_time = datetime.now(st.session_state.agent.timezone)
                st.info(f"🕒 Timezone: {st.session_state.agent.timezone} ({current_time.strftime('%Z')})")
            except:
                st.info("🕒 Timezone: UTC")
        else:
            st.error("🔴 Agent: Offline")
        
        st.subheader("🛠️ Available Tools")
        if st.session_state.agent:
            tools_info = {
                "retrieve_documents": "📄 Document Search",
                "summarize_document": "📋 Document Summary",
                "web_search": "🌐 Web Search",
                "wikipedia_search": "📚 Wikipedia",
                "calculator": "🧮 Calculator",
                "get_datetime": "🕒 Date & Time",
                "world_clock": "🌍 World Clock"
            }
            
            for tool in st.session_state.agent.tools:
                tool_display = tools_info.get(tool.name, f"🔧 {tool.name}")
                st.write(f"• {tool_display}")
        else:
            st.write("Initialize agent to see tools")
        
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "Summarize the document",
            "What is the main topic?",
            "What's the current time?",
            "What time is it in Tokyo?",
            "Show time in major cities",
            "Search for recent AI news",
            "Calculate 25 * 4 + 100",
            "Find info about machine learning"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{hash(query)}", use_container_width=True):
                if st.session_state.agent:
                    with st.spinner("Processing..."):
                        response = st.session_state.agent.run(query)
                    st.session_state.chat_history.append((query, response))
                    st.rerun()
                else:
                    st.error("Please initialize agent first")
        
        st.subheader("💡 Tips")
        st.markdown("""
        • **Load documents** first for document analysis
        • **Ask specific questions** for better results
        • **Use time queries** like "time in London" or "current date"
        • **Try calculations** with the calculator tool
        • **Use web search** for current information
        • **Be patient** - complex queries take time
        """)
    
    st.markdown("---")
    st.markdown(
        "🚀 **RAG ReAct Agent** | "
        "Built with Streamlit • Powered by Groq • Enhanced with LangChain & LangGraph"
    )

if __name__ == "__main__":
    main()