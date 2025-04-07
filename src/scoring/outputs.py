from typing import List

from pydantic import BaseModel, Field


class ExtractedOutput(BaseModel):
    indices: List[int]


class Result(BaseModel):
    quotes: List[str] = Field(default_factory=list)
    sentiment_scores: List[int] = Field(default_factory=list)


class FilteredWithSentimentQuotesOutput(BaseModel):
    """
    A Pydantic model representing filtered quotes based on relevance.

    Attributes:
        related_indices (List[int]): Indices of the quotes to include in the final selection.
        sentiment_scores (List[int]): The numerical score assigned based on the input quotes and criteria.
                     Possible values:
                     -1: Negative sentiment
                      0: Neutral sentiment
                      1: Positive sentiment
    """
    related_indices: List[int]
    sentiment_scores: List[int]


class ScoredOverallOutput(BaseModel):
    """
    A Pydantic model to represent the output of the scoring function.

    Attributes:
        reason (str): The rationale or explanation behind the assigned score.
    """
    reason: str


class ScoredThemeOutput(BaseModel):
    """
    A Pydantic model representing the output of the theme relevance scoring function.

    Attributes:
        reason (str): Detailed explanation of why the assigned score was given, 
                      referencing the analyzed quotes and criteria.
        relevance_score (int): A numerical score representing the company's relevance to the theme:
                     - 0: Not at all relevant to the theme
                     - 1: Slightly relevant
                     - 2: Highly relevant
        sentiment_score (int): A numerical score representing the company's sentiment to the theme:
                     - -1: Negative
                     - 0: Neutral
                     - 1: Positive
    """
    reason: str
    relevance_score: int
    sentiment_score: int
