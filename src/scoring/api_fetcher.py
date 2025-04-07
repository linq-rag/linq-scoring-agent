from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

import dotenv
import groq
import httpx
import openai
from groq.types.chat import ChatCompletion as GroqChatCompletion
from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from openai.types.chat import ChatCompletionChunk as OpenAIChatCompletionChunk
from openai.types.chat import ParsedChatCompletion as OpenAIParsedChatCompletion
from pydantic import BaseModel

from .utils import retry_fetch

BaseModelType = Type[BaseModel]

dotenv.load_dotenv()

__all__ = [
    "OpenAIAPIFetcher",
    "AsyncOpenAIAPIFetcher",
    "GroqAPIFetcher",
    "AsyncGroqAPIFetcher",
    "FuriosaAPIFetcher",
    "AsyncFuriosaAPIFetcher",
]

# ----------------------------------------
# TypeVar definitions (from your original question)
# ----------------------------------------
_SyncClientT = TypeVar(
    "_SyncClientT",
    bound=Union[httpx.Client, openai.OpenAI, groq.Groq]
)
_AsyncClientT = TypeVar(
    "_AsyncClientT",
    bound=Union[httpx.AsyncClient, openai.AsyncOpenAI, groq.AsyncGroq]
)
_ClientT = TypeVar("_ClientT", bound=Union[httpx.Client, openai.OpenAI, groq.Groq, httpx.AsyncClient, openai.AsyncOpenAI, groq.AsyncGroq])


# ===================================================
# 1) BaseAPIFetcher
#    - Abstract base class for both sync/async fetchers
# ===================================================
class BaseAPIFetcher(Generic[_ClientT], ABC):
    """
    Abstract base class for all Fetchers.
    Common functionalities shared by both sync and async fetchers can be defined here.
    """

    def __init__(self, client: Union[_SyncClientT, _AsyncClientT]):
        self.client = client

    @abstractmethod
    def close(self) -> None:
        """Method to handle resource cleanup."""


# ===================================================
# 2) SyncAPIFetcher
#    - Base class for synchronous fetchers (httpx.Client, openai.OpenAI, groq.Groq)
# ===================================================
class SyncAPIFetcher(BaseAPIFetcher[_SyncClientT], ABC):
    """
    Base class for all synchronous fetchers.
    Supports clients like httpx.Client, openai.OpenAI, and groq.Groq.
    """

    def __init__(self, client: _SyncClientT):
        super().__init__(client)

    @abstractmethod
    def fetch_chat_completion(self, url: str) -> str:
        """Method to send a synchronous request or perform some synchronous operation."""

    def close(self) -> None:
        """Handles resource cleanup for synchronous clients."""
        if hasattr(self.client, "close"):
            self.client.close()


# ===================================================
# 3) AsyncAPIFetcher
#    - Base class for asynchronous fetchers (httpx.AsyncClient, openai.AsyncOpenAI, groq.AsyncGroq)
# ===================================================
class AsyncAPIFetcher(BaseAPIFetcher[_AsyncClientT], ABC):
    """
    Base class for all asynchronous fetchers.
    Supports clients like httpx.AsyncClient, openai.AsyncOpenAI, and groq.AsyncGroq.
    """

    def __init__(self, client: _AsyncClientT):
        super().__init__(client)

    @abstractmethod
    async def fetch_chat_completion(self, url: str) -> str:
        """Method to send an asynchronous request or perform some async operation."""

    def close(self) -> None:
        """Handles resource cleanup for asynchronous clients."""
        if hasattr(self.client, "close"):
            self.client.close()


# ===================================================
# 4) OpenAIAPIFetcher (Synchronous)
# ===================================================
class OpenAIAPIFetcher(SyncAPIFetcher[openai.OpenAI]):
    """
    Synchronous Fetcher specialized for OpenAI's API.
    """

    def __init__(self, client: Optional[openai.OpenAI] = None):
        if client is None:
            client = openai.OpenAI()
        super().__init__(client)

    retry_fetch(0.01, 1)
    def fetch_chat_completion(self, **kwargs) -> OpenAIChatCompletion | Stream[OpenAIChatCompletionChunk]:
        """Example fetch method simulating a request to OpenAI."""

        return self.client.chat.completions.create(**kwargs)

    retry_fetch(0.01, 1)
    def fetch_parsed_completion(self, **kwargs) -> OpenAIParsedChatCompletion:
        """Example fetch method simulating a request to OpenAI."""

        return self.client.beta.chat.completions.parse(**kwargs)

    retry_fetch(0.01, 1)
    def fetch_parsed_output(self, content: str, response_format: BaseModelType) -> OpenAIParsedChatCompletion:
        return self.client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Given the following data, format it with the given response format: {content}.
If it is not possible, return an empty value with the given response format.
"""
                }
            ],
            temperature=0.0,
            response_format=response_format,
        )


# ===================================================
# 5) AsyncOpenAIAPIFetcher (Asynchronous)
# ===================================================
class AsyncOpenAIAPIFetcher(AsyncAPIFetcher[openai.AsyncOpenAI]):
    """
    Asynchronous Fetcher specialized for OpenAI's API.
    """

    def __init__(self, client: Optional[openai.AsyncOpenAI] = None):
        if client is None:
            client = openai.AsyncOpenAI()
        super().__init__(client)

    retry_fetch(0.01, 1)
    async def fetch_chat_completion(self, **kwargs) -> OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]:
        """Example fetch method simulating a request to OpenAI."""

        return await self.client.chat.completions.create(**kwargs)

    retry_fetch(0.01, 1)
    async def fetch_parsed_completion(self, **kwargs) -> OpenAIParsedChatCompletion:
        """Example fetch method simulating a request to OpenAI."""

        return await self.client.beta.chat.completions.parse(**kwargs)

    retry_fetch(0.01, 1)
    async def fetch_parsed_output(self, content: str, response_format: BaseModelType) -> OpenAIParsedChatCompletion:
        return await self.client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Given the following data, format it with the given response format: {content}.
If it is not possible, return an empty value with the given response format.
"""
                }
            ],
            temperature=0.0,
            response_format=response_format,
        )


# ===================================================
# 6) GroqAPIFetcher (Synchronous)
# ===================================================
class GroqAPIFetcher(SyncAPIFetcher[groq.Groq]):
    """
    Synchronous Fetcher specialized for Groq's API.
    """

    def __init__(self, client: Optional[groq.Groq] = None):
        if client is None:
            client = groq.Groq()
        super().__init__(client)

    retry_fetch(0.01, 1)
    def fetch_chat_completion(self, **kwargs) -> GroqChatCompletion | Stream[GroqChatCompletionChunk]:
        """Example fetch method simulating a request to OpenAI."""

        return self.client.chat.completions.create(**kwargs)


# ===================================================
# 7) AsyncGroqAPIFetcher (Asynchronous)
# ===================================================
class AsyncGroqAPIFetcher(AsyncAPIFetcher[groq.AsyncGroq]):
    """
    Asynchronous Fetcher specialized for Groq's API.
    """

    def __init__(self, client: Optional[groq.AsyncGroq] = None):
        if client is None:
            client = groq.AsyncGroq()
        super().__init__(client)

    retry_fetch(0.01, 1)
    async def fetch_chat_completion(self, **kwargs) -> GroqChatCompletion | Stream[GroqChatCompletionChunk]:
        """Example fetch method simulating a request to OpenAI."""

        return await self.client.chat.completions.create(**kwargs)


# ===================================================
# 8) SyncAPIFetcher
#    - Base class for synchronous fetchers (httpx.Client, openai.OpenAI, groq.Groq)
# ===================================================
class FuriosaAPIFetcher(SyncAPIFetcher[httpx.Client]):
    """
    Base class for all synchronous fetchers.
    Supports clients like httpx.Client, openai.OpenAI, and groq.Groq.
    """

    def __init__(
            self,
            client: Optional[httpx.Client] = None,
            url: str = "http://k8s.eval.furiosa.in:31780/v1/chat/completions",
            headers: Dict[str, str] = None,
    ):
        if client is None:
            client = httpx.Client()

        if headers is None:
            headers = {"Content-Type": "application/json"}

        self.headers = headers
        self.url = url

        super().__init__(client)

    retry_fetch(0.01, 1)
    def fetch_chat_completion(self, **kwargs) -> Dict[str, Any]:
        """Method to send a synchronous request or perform some synchronous operation."""
        response = self.client.post(url=self.url, headers=self.headers, data=kwargs)
        return response.json()


# ===================================================
# 9) AsyncAPIFetcher
#    - Base class for asynchronous fetchers (httpx.AsyncClient, openai.AsyncOpenAI, groq.AsyncGroq)
# ===================================================
class AsyncFuriosaAPIFetcher(AsyncAPIFetcher[httpx.AsyncClient], ABC):
    """
    Base class for all asynchronous fetchers.
    Supports clients like httpx.AsyncClient, openai.AsyncOpenAI, and groq.AsyncGroq.
    """

    def __init__(
            self,
            client: Optional[httpx.AsyncClient] = None,
            url: str = "http://k8s.eval.furiosa.in:31780/v1/chat/completions",
            headers: Dict[str, str] = None,
    ):
        if client is None:
            client = httpx.AsyncClient()

        if headers is None:
            headers = {"Content-Type": "application/json"}

        self.headers = headers
        self.url = url

        super().__init__(client)

    retry_fetch(0.01, 1)
    async def fetch_chat_completion(self, **kwargs) -> Dict[str, Any]:
        """Method to send an asynchronous request or perform some async operation."""
        response = await self.client.post(url=self.url, headers=self.headers, data=kwargs)
        return await response.json()
