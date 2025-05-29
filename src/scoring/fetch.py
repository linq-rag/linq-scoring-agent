"""
LINQ Scoring Agent - LLM API Fetching Module

This module handles asynchronous communication with multiple LLM providers (OpenAI, Groq, Furiosa)
for extracting and filtering financial themes from earnings call transcripts. It implements
batch processing with parallel execution and error handling.

Key Features:
- Multi-provider LLM API support (OpenAI, Groq, Furiosa)
- Asynchronous batch processing for performance
- Structured output parsing with Pydantic models
- Automatic text chunking for large transcripts
- Usage tracking and error handling
"""

import asyncio
import logging
import traceback
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from ._default import DEFAULT_FURIOSA_KWARGS, DEFAULT_GROQ_KWARGS, DEFAULT_OPENAI_KWARGS
from .api_fetcher import (
    AsyncFuriosaAPIFetcher,
    AsyncGroqAPIFetcher,
    AsyncOpenAIAPIFetcher,
)
from .messages import (
    get_overall_extracting_messages,
    get_overall_filtering_messages,
    get_theme_extracting_messages,
    get_theme_filtering_messages,
)
from .outputs import ExtractedOutput, FilteredWithSentimentQuotesOutput, Result
from .utils import get_sentences, split_list_into_n, split_transcript_into_n

BaseModelType = Type[BaseModel]

logger = logging.getLogger(__name__)

# Initialize API fetchers for different LLM providers
openai_async_fetcher = AsyncOpenAIAPIFetcher()
groq_async_fetcher = AsyncGroqAPIFetcher()
furiosa_async_fetcher = AsyncFuriosaAPIFetcher()


async def fetch_parsed(
        messages: List[Dict[str, str]],
        response_format: BaseModelType,
        fetch_type: Literal["groq", "furiosa", "openai"],
) -> Tuple[BaseModelType | None, Dict[str, Dict]]:
    """
    Fetch parsed structured output from specified LLM provider.
    
    This function routes requests to different LLM providers and ensures
    consistent structured output format. For non-OpenAI providers, it
    uses a two-step process: first generation, then parsing.
    
    Args:
        messages: List of message dictionaries for the chat completion
        response_format: Pydantic model class defining expected output structure
        fetch_type: Which LLM provider to use for generation
        
    Returns:
        Tuple containing:
        - Parsed model instance or None if parsing failed
        - Usage statistics dictionary with provider-specific metrics
        
    Note:
        Groq and Furiosa providers require additional OpenAI parsing step
        for structured output format compliance
    """
    if fetch_type == "groq":
        kwargs = DEFAULT_GROQ_KWARGS | {"messages": messages}
        llama_chat_completion = await groq_async_fetcher.fetch_chat_completion(**kwargs)
    elif fetch_type == "furiosa":
        kwargs = DEFAULT_FURIOSA_KWARGS | {"messages": messages}
        llama_chat_completion = await furiosa_async_fetcher.fetch_chat_completion(**kwargs)
    else:
        # Direct OpenAI structured output
        kwargs = DEFAULT_OPENAI_KWARGS | {"messages": messages, "response_format": response_format}
        openai_parsed_completion = await openai_async_fetcher.fetch_parsed_completion(**kwargs)
        extracted_output = openai_parsed_completion.choices[0].message.parsed
        usage = {"openai": openai_parsed_completion.usage.model_dump()}
        return extracted_output, usage

    # Parse non-OpenAI responses using OpenAI structured parser
    openai_parsed_completion = await openai_async_fetcher.fetch_parsed_output(
        content=llama_chat_completion.choices[0].message.content,
        response_format=response_format,
    )

    parsed_output = openai_parsed_completion.choices[0].message.parsed
    if llama_chat_completion.usage is not None:
        llama_usage = llama_chat_completion.usage.model_dump()
    else:
        llama_usage = None
    usage = {
        f"{fetch_type}": llama_usage, "openai": openai_parsed_completion.usage.model_dump()
    }

    return parsed_output, usage


async def _fetch_extracted_output(
        company_name: str,
        text: str,
        extraction_type: Literal["theme", "overall"],
        fetch_type: Literal["groq", "furiosa", "openai"],
        theme: Optional[str] = None,
        max_size: int = 20,
) -> tuple[Result, list[Dict[str, Dict]]]:
    """
    Extract relevant quotes from a single transcript chunk.
    
    This internal function processes one chunk of transcript text to identify
    and extract quotes relevant to either overall sentiment or specific themes.
    It uses sentence-level chunking for granular processing.
    
    Args:
        company_name: Name of the company for context
        text: Transcript text chunk to process
        extraction_type: Whether to extract "theme" or "overall" content
        fetch_type: LLM provider to use
        theme: Specific theme description (required if extraction_type is "theme")
        max_size: Maximum number of sentences per sub-chunk
        
    Returns:
        Tuple containing:
        - Result object with extracted quotes
        - List of usage statistics from API calls
        
    The function automatically handles sentence segmentation and creates
    appropriately formatted prompts for the LLM provider.
    """
    lines = get_sentences(text=text)
    text_dict = {i: line for i, line in enumerate(lines)}

    # Create sub-chunks from sentences for processing
    text_chunks = [
        {i: text_dict[i] for i in range(start, min(start + max_size, len(lines)))}
        for start in range(0, len(lines), max_size)
    ]

    async def fetch_chunk(chunk_text_dict: Dict[int, str]):
        """Process individual text chunk for quote extraction."""
        formatted_text = "\n".join([f"**Quote {key}**. {value.strip()}" for key, value in chunk_text_dict.items()])

        # Select appropriate prompt based on extraction type
        if extraction_type == "theme":
            if theme is None:
                raise ValueError("No theme specified")
            else:
                messages = get_theme_extracting_messages(company_name, theme, formatted_text)
        else:
            messages = get_overall_extracting_messages(company_name, formatted_text)
        print(messages, file=open("extract.txt", "w"))
        try:
            extracted_output, _usage = await fetch_parsed(
                messages=messages, response_format=ExtractedOutput, fetch_type=fetch_type
            )

            # Extract quotes based on returned indices
            _quotes = [chunk_text_dict[i] for i in extracted_output.indices if i in chunk_text_dict]

            return _quotes, _usage
        except Exception as e:
            print(f"An Error occurred while processing extracting Theme {theme}: {str(e)[:30]}")
            traceback.print_exc()
            return [], {}

    # Process all chunks in parallel
    results = await asyncio.gather(*[fetch_chunk(chunk) for chunk in text_chunks])

    # Aggregate results from all chunks
    final_result = Result()
    final_usage = []

    for q, u in results:
        try:
            if q:  # Only extend if q is not empty
                final_result.quotes.extend(q)
            final_usage.append(u)
        except Exception:
            continue

    return final_result, final_usage


async def _fetch_filtered_output(
        company_name: str,
        quotes: List[str],
        extraction_type: Literal["theme", "overall"],
        fetch_type: Literal["groq", "furiosa", "openai"],
        theme: Optional[str] = None,
        max_size: int = 20,
) -> tuple[Result, list[Dict[str, Dict]]]:
    """
    Filter extracted quotes for relevance and assign sentiment scores.
    
    This internal function takes previously extracted quotes and filters them
    for relevance to the specified theme or overall sentiment. It also assigns
    sentiment scores to each relevant quote.
    
    Args:
        company_name: Name of the company for context
        quotes: List of previously extracted quotes to filter
        extraction_type: Whether filtering for "theme" or "overall" content
        fetch_type: LLM provider to use (note: filtering always uses OpenAI)
        theme: Specific theme description (required if extraction_type is "theme")
        max_size: Maximum number of quotes per chunk
        
    Returns:
        Tuple containing:
        - Result object with filtered quotes and sentiment scores
        - List of usage statistics from API calls
        
    Note:
        Filtering pipeline always uses OpenAI for consistent sentiment scoring,
        regardless of the fetch_type parameter
    """
    # Split quotes into manageable chunks
    quote_chunks = [
        quotes[i:i + max_size] for i in range(0, len(quotes), max_size)
    ]

    async def fetch_chunk(chunk_quotes: List[str]):
        """Process individual quote chunk for filtering and sentiment analysis."""
        formatted_quotes = "\n".join([f"**Quotes {i}**. {value.strip()}" for i, value in enumerate(chunk_quotes)])

        # Select appropriate prompt based on extraction type
        if extraction_type == "theme":
            if theme is None:
                raise ValueError("No theme specified")
            else:
                messages = get_theme_filtering_messages(company_name, theme, formatted_quotes)
        else:
            messages = get_overall_filtering_messages(company_name, formatted_quotes)
        print(messages, file=open("filter.txt", "w"))
        try:
            # Note: Filtering always uses OpenAI for consistent sentiment scoring
            filtered_output, _usage = await fetch_parsed(
                messages=messages, response_format=FilteredWithSentimentQuotesOutput, fetch_type="openai"
            )
            
            # Extract filtered quotes and their sentiment scores
            _filtered_quotes = [chunk_quotes[i] for i in filtered_output.related_indices if i < len(chunk_quotes)]
            _filtered_sentiments = [i for i in filtered_output.sentiment_scores]

            return _filtered_quotes, _filtered_sentiments, _usage
        except Exception as e:
            print(f"An Error occurred while processing filtering Theme {theme}: {str(e)[:30]}")
            traceback.print_exc()
            return [], [], {}

    # Process all quote chunks in parallel
    results = await asyncio.gather(*[fetch_chunk(q) for q in quote_chunks])

    # Aggregate filtered results
    final_result = Result()
    final_usage = []

    for q, s, u in results:
        if q:  # Only extend if q is not empty
            final_result.quotes.extend(q)
        if s:  # Only extend if s is not empty
            final_result.sentiment_scores.extend(s)
        final_usage.append(u)

    return final_result, final_usage


async def fetch_extracted_output(
        company_name: str,
        text: str,
        extraction_type: Literal["theme", "overall"],
        fetch_type: Literal["groq", "furiosa", "openai"],
        theme: Optional[str] = None,
        num_split: int = 40,
) -> Tuple[Result, List[Dict[str, Dict]]]:
    """
    Extract relevant quotes from full earnings call transcript.
    
    This is the main public interface for quote extraction. It handles large
    transcripts by splitting them into smaller chunks and processing them
    in parallel for better performance and API compliance.
    
    Args:
        company_name: Name of the company for contextual prompts
        text: Full earnings call transcript text
        extraction_type: Extract "theme"-specific or "overall" sentiment content
        fetch_type: LLM provider to use ("groq", "furiosa", "openai")
        theme: Theme description (required when extraction_type is "theme")
        num_split: Number of chunks to split the transcript into
        
    Returns:
        Tuple containing:
        - Result object with all extracted quotes
        - List of usage statistics from all API calls
        
    The function automatically handles transcript segmentation, parallel processing,
    and result aggregation with comprehensive error handling.
    """
    # Split transcript into manageable chunks
    split_texts = split_transcript_into_n(text=text, n=num_split)

    # Create partial function with common parameters
    fetch_partial = partial(
        _fetch_extracted_output,
        company_name=company_name,
        extraction_type=extraction_type,
        fetch_type=fetch_type,
        theme=theme
    )

    # Process all text chunks in parallel with progress tracking
    tasks = [fetch_partial(text=text_chunk) for text_chunk in split_texts]
    results = await tqdm_asyncio.gather(*tasks, desc="fetching extraction", leave=False)

    # Aggregate results from all chunks
    usages = []
    result = Result()
    for output, usage in results:
        try:
            result.quotes.extend(output.quotes)
            usages.append(usage)
        except AttributeError as e:
            print(f"An `AttributeError` occurred while processing `fetch_extracted_output`: {e}")
            traceback.print_exc()
            continue
    return result, usages


async def fetch_filtered_output(
        company_name: str,
        quotes: List[str],
        extraction_type: Literal["theme", "overall"],
        fetch_type: Literal["groq", "furiosa", "openai"],
        theme: Optional[str] = None,
        num_split: int = 20,
) -> Tuple[Result, List[Dict[str, Dict]]]:
    """
    Filter extracted quotes for relevance and assign sentiment scores.
    
    This is the main public interface for quote filtering. It takes previously
    extracted quotes and filters them for relevance while assigning sentiment
    scores. Large quote lists are processed in parallel chunks.
    
    Args:
        company_name: Name of the company for contextual prompts
        quotes: List of previously extracted quotes to filter
        extraction_type: Filter for "theme"-specific or "overall" sentiment
        fetch_type: LLM provider preference (note: filtering uses OpenAI)
        theme: Theme description (required when extraction_type is "theme")
        num_split: Number of chunks to split the quotes into
        
    Returns:
        Tuple containing:
        - Result object with filtered quotes and sentiment scores
        - List of usage statistics from all API calls
        
    Note:
        The filtering pipeline always uses OpenAI for consistent sentiment scoring,
        regardless of the fetch_type parameter specified.
    """
    # Split quotes into manageable chunks
    split_quotes = split_list_into_n(lst=quotes, n=num_split)

    # Create partial function with common parameters
    fetch_partial = partial(
        _fetch_filtered_output, company_name=company_name, extraction_type=extraction_type, fetch_type=fetch_type,
        theme=theme
    )

    # Process all quote chunks in parallel with progress tracking
    tasks = [fetch_partial(quotes=_quotes) for _quotes in split_quotes]
    results = await tqdm_asyncio.gather(*tasks, desc="fetching filtering", leave=False)

    # Aggregate filtered results
    usages = []
    result = Result()
    for output, usage in results:
        try:
            result.quotes.extend(output.quotes)
            result.sentiment_scores.extend(output.sentiment_scores)
            usages.append(usage)
        except AttributeError as e:
            print(f"An `AttributeError` occurred while processing `fetch_extracted_output`: {e}")
            continue
    return result, usages
