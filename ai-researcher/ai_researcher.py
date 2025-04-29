import re

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from duckduckgo_search import DDGS

import requests
from bs4 import BeautifulSoup

summary_template = """
Summarize the following content into a concise paragraph that directly addresses the query. Ensure the summary
highlights the key points relevant to the query while maintaining clarity and completeness.
Query: {query}
Content: {content}
"""

generate_response_template = """
Given the following user query and content, generate a response that directly answers the query using relevant
information from the content. Ensure that the response is clear, concise, and well-structured.
Additionally, provide a brief summary of the key points from the response.
Question: {question}
Context: {context}
Answer:
"""

class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    summarized_results: list[str]
    response: str

class ResearchStateInput(TypedDict):
    query: str

class ResearchStateOutput(TypedDict):
    sources: list[str]
    response: str

def search_web_tavily(state: ResearchState):
    search = TavilySearchResults(max_results=3)
    search_results = search.invoke(state["query"])

    return  {
        "sources": [result['url'] for result in search_results],
        "web_results": [result['content'] for result in search_results]
    }

def search_web_DDGS(state: ResearchState):
    # results = DDGS(state["query"], max_results=3)
    with DDGS() as ddgs:
        results = ddgs.text("your query here")
        print(results)
    return {
        "sources": [result['href'] for result in results],
        "web_results": [result['body'] for result in results]
    }

def search_web(state: ResearchState):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(f"https://www.google.com/search?q={state['query']}", headers=headers)

    # Debug: Print the raw HTML to verify structure
    print(response.text)

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract search results
    results = soup.select("div.bNg8Rb")  # Updated selector for Google search results
    if not results:
        print("No results found. Check the HTML structure or if Google blocked the request.")

    sources = []
    web_results = []

    for result in results:
        link = result.select_one("a")  # Extract the link
        snippet = result.select_one(".VwiC3b")  # Extract the snippet

        if link and snippet:
            sources.append(link["href"])
            web_results.append(snippet.text)

    return {
        "sources": sources,
        "web_results": web_results
    }

def summarize_results(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model

    summarized_results = []
    for content in state["web_results"]:
        summary = chain.invoke({"query": state["query"], "content": content})
        clean_content = clean_text(summary.content)
        summarized_results.append(clean_content)

    return {
        "summarized_results": summarized_results
    }

def generate_response(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model

    content = "\n\n".join([summary for summary in state["summarized_results"]])

    return {
        "response": chain.invoke({"question": state["query"], "context": content})
    }

def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

builder = StateGraph(
    ResearchState,
    input=ResearchStateInput,
    output=ResearchStateOutput
)

builder.add_node("search_web", search_web)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "search_web")
builder.add_edge("search_web", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

st.title("AI Researcher")
query = st.text_input("Enter your research query:")

if query:
    response_state = graph.invoke({"query": query})
    st.write(clean_text(response_state["response"].content))

    st.subheader("Sources:")
    for source in response_state["sources"]:
        st.write(source)