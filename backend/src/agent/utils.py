import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a web search using DuckDuckGo search (no API key required).
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of dictionaries containing title, url, and snippet
    """
    search_results = []
    
    # Use DuckDuckGo instant answer API
    ddg_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(ddg_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        
        # Get related topics if available
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics'][:num_results]:
                if isinstance(topic, dict) and 'Text' in topic and 'FirstURL' in topic:
                    search_results.append({
                        'title': topic.get('Text', '')[:100] + '...' if len(topic.get('Text', '')) > 100 else topic.get('Text', ''),
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', '')
                    })
        
        # If we don't have enough results, try to get the abstract
        if len(search_results) < num_results and 'Abstract' in data and data['Abstract']:
            search_results.append({
                'title': data.get('Heading', query),
                'url': data.get('AbstractURL', ''),
                'snippet': data.get('Abstract', '')
            })
    
    # If DuckDuckGo doesn't return enough results, create a fallback result
    if len(search_results) == 0:
        search_results.append({
            'title': f"Search results for: {query}",
            'url': f"https://duckduckgo.com/?q={quote_plus(query)}",
            'snippet': f"Please search for '{query}' to find relevant information."
        })
    
    return search_results[:num_results]


def resolve_urls(search_results: List[Dict[str, str]], id: int) -> Dict[str, str]:
    """
    Create a map of the original URLs to short URLs with unique IDs.
    """
    prefix = f"https://search.result.com/id/"
    resolved_map = {}
    
    for idx, result in enumerate(search_results):
        url = result.get('url', '')
        if url and url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"
    
    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on citations.
    
    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries containing citation info
        
    Returns:
        str: The text with citation markers inserted.
    """
    if not citations_list:
        return text
    
    # For OpenAI, we'll append citations at the end of the text
    citation_markers = ""
    for citation in citations_list:
        for segment in citation.get("segments", []):
            citation_markers += f" [{segment['label']}]({segment['short_url']})"
    
    return text + citation_markers


def get_citations_from_search_results(search_results: List[Dict[str, str]], resolved_urls_map: Dict[str, str]) -> List[Dict]:
    """
    Create citations from search results for OpenAI compatibility.
    
    Args:
        search_results: List of search result dictionaries
        resolved_urls_map: Map of original URLs to short URLs
        
    Returns:
        List of citation dictionaries
    """
    citations = []
    
    for idx, result in enumerate(search_results):
        url = result.get('url', '')
        title = result.get('title', f'Source {idx + 1}')
        
        if url and url in resolved_urls_map:
            citation = {
                "start_index": 0,
                "end_index": len(result.get('snippet', '')),
                "segments": [{
                    "label": title.split('.')[0] if '.' in title else title,
                    "short_url": resolved_urls_map[url],
                    "value": url
                }]
            }
            citations.append(citation)
    
    return citations


# Keep the old function for backward compatibility but mark as deprecated
def get_citations(response, resolved_urls_map):
    """
    Legacy function for Gemini compatibility - now returns empty list.
    Use get_citations_from_search_results for OpenAI implementation.
    """
    return []
