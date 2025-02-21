import os
import random
import requests
import vertexai
import json
from typing import TypedDict, Literal
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from curriculums import get_curriculum 
from search import search_latest_resource 
from book import recommend_book 

from google.cloud import pubsub_v1


PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env




project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env

# Connect to resourse needed from Google Cloud
llm = ChatVertexAI(model_name="gemini-1.5-flash-001",location="us-west1")

