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
openai_async_fetcher = AsyncOpenAIAPIFetcher()
groq_async_fetcher = AsyncGroqAPIFetcher()
furiosa_async_fetcher = AsyncFuriosaAPIFetcher()


async def fetch_parsed(
        messages: List[Dict[str, str]],
        response_format: BaseModelType,
        fetch_type: Literal["groq", "furiosa", "openai"],
) -> Tuple[BaseModelType | None, Dict[str, Dict]]:

    if fetch_type == "groq":
        kwargs = DEFAULT_GROQ_KWARGS | {"messages": messages}
        llama_chat_completion = await groq_async_fetcher.fetch_chat_completion(**kwargs)
    elif fetch_type == "furiosa":
        kwargs = DEFAULT_FURIOSA_KWARGS | {"messages": messages}
        llama_chat_completion = await furiosa_async_fetcher.fetch_chat_completion(**kwargs)
    else:
        kwargs = DEFAULT_OPENAI_KWARGS | {"messages": messages, "response_format": response_format}
        openai_parsed_completion = await openai_async_fetcher.fetch_parsed_completion(**kwargs)
        extracted_output = openai_parsed_completion.choices[0].message.parsed
        usage = {"openai": openai_parsed_completion.usage.model_dump()}
        return extracted_output, usage

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
    lines = get_sentences(text=text)
    text_dict = {i: line for i, line in enumerate(lines)}

    text_chunks = [
        {i: text_dict[i] for i in range(start, min(start + max_size, len(lines)))}
        for start in range(0, len(lines), max_size)
    ]

    async def fetch_chunk(chunk_text_dict: Dict[int, str]):
        formatted_text = "\n".join([f"**Quote {key}**. {value.strip()}" for key, value in chunk_text_dict.items()])

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

            _quotes = [chunk_text_dict[i] for i in extracted_output.indices if i in chunk_text_dict]

            return _quotes, _usage
        except Exception as e:
            print(f"An Error occurred while processing extracting Theme {theme}: {str(e)[:30]}")
            traceback.print_exc()
            return [], {}

    results = await asyncio.gather(*[fetch_chunk(chunk) for chunk in text_chunks])

    final_result = Result()
    final_usage = []

    for q, u in results:
        try:
            final_result.quotes.extend(q)
            final_usage.append(u)
        except:
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
    
    quote_chunks = [
        quotes[i:i + max_size] for i in range(0, len(quotes), max_size)
    ]

    async def fetch_chunk(chunk_quotes: List[str]):
        formatted_quotes = "\n".join([f"**Quotes {i}**. {value.strip()}" for i, value in enumerate(chunk_quotes)])

        if extraction_type == "theme":
            if theme is None:
                raise ValueError("No theme specified")
            else:
                messages = get_theme_filtering_messages(company_name, theme, formatted_quotes)
        else:
            messages = get_overall_filtering_messages(company_name, formatted_quotes)
        print(messages, file=open("filter.txt", "w"))
        try:
            filtered_output, _usage = await fetch_parsed(
                messages=messages, response_format=FilteredWithSentimentQuotesOutput, fetch_type="openai"
            )
            _filtered_quotes = [chunk_quotes[i] for i in filtered_output.related_indices if i < len(chunk_quotes)]
            _filtered_sentiments = [i for i in filtered_output.sentiment_scores]

            return _filtered_quotes, _filtered_sentiments, _usage
        except Exception as e:
            print(f"An Error occurred while processing filtering Theme {theme}: {str(e)[:30]}")
            traceback.print_exc()
            return [], [], {}

    results = await asyncio.gather(*[fetch_chunk(q) for q in quote_chunks])

    # 결과 합치기
    final_result = Result()
    final_usage = []

    for q, s, u in results:
        final_result.quotes.extend(q)
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

    split_texts = split_transcript_into_n(text=text, n=num_split)

    fetch_partial = partial(
        _fetch_extracted_output,
        company_name=company_name,
        extraction_type=extraction_type,
        fetch_type=fetch_type,
        theme=theme
    )


    # 키워드 인자 대신 위치 인자로 전달
    tasks = [fetch_partial(text=text_chunk) for text_chunk in split_texts]
    results = await tqdm_asyncio.gather(*tasks, desc="fetching extraction", leave=False)

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

    split_quotes = split_list_into_n(lst=quotes, n=num_split)

    fetch_partial = partial(
        _fetch_filtered_output, company_name=company_name, extraction_type=extraction_type, fetch_type=fetch_type,
        theme=theme
    )

    tasks = [fetch_partial(quotes=_quotes) for _quotes in split_quotes]

    results = await tqdm_asyncio.gather(*tasks, desc="fetching filtering", leave=False)

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
