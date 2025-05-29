"""
LINQ Scoring Agent - Output Data Models Module

This module defines Pydantic data models for structured outputs from the
LLM-based extraction, filtering, and scoring pipelines. These models ensure
consistent data formats and validation across the entire processing pipeline.

Key Models:
- ExtractedOutput: Indices of extracted sentences
- Result: Quotes and sentiment scores container
- FilteredWithSentimentQuotesOutput: Filtered quotes with sentiment analysis
- ScoredOverallOutput: Overall sentiment scoring results
- ScoredThemeOutput: Theme-specific relevance and sentiment scoring
"""

from typing import List

from pydantic import BaseModel, Field


class ExtractedOutput(BaseModel):
    """
    Output format for sentence extraction operations.
    
    This model represents the result of extracting relevant sentences
    from earnings call transcripts, containing only the indices of
    sentences that match the extraction criteria.
    
    Attributes:
        indices: List of sentence indices that were extracted as relevant
    """
    indices: List[int]


class Result(BaseModel):
    """
    Container for processed quotes and their sentiment scores.
    
    This is the primary data structure for storing extracted and filtered
    results throughout the pipeline. It maintains parallel lists of quotes
    and their corresponding sentiment scores.
    
    Attributes:
        quotes: List of extracted text quotes from transcripts
        sentiment_scores: List of integer sentiment scores (-1, 0, 1)
                         corresponding to each quote
    """
    quotes: List[str] = Field(default_factory=list)
    sentiment_scores: List[int] = Field(default_factory=list)


class FilteredWithSentimentQuotesOutput(BaseModel):
    """
    Output format for quote filtering and sentiment analysis operations.
    
    This model represents the result of filtering extracted quotes for
    relevance and assigning sentiment scores. It contains indices pointing
    to the relevant quotes and their corresponding sentiment values.
    
    Attributes:
        related_indices: Indices of quotes deemed relevant for the analysis
        sentiment_scores: Sentiment scores for each relevant quote:
                         -1 = Negative sentiment
                          0 = Neutral sentiment  
                          1 = Positive sentiment
    """
    related_indices: List[int]
    sentiment_scores: List[int]


class ScoredOverallOutput(BaseModel):
    """
    Output format for overall sentiment scoring operations.
    
    This model represents the result of scoring a company's overall
    sentiment based on filtered quotes from earnings calls. It provides
    human-readable reasoning for the assigned score.
    
    Attributes:
        reason: Detailed explanation of the scoring rationale,
                referencing specific quotes and analysis criteria
    """
    reason: str


class ScoredThemeOutput(BaseModel):
    """
    Output format for theme-specific relevance and sentiment scoring.
    
    This model represents the comprehensive analysis of a company's
    relationship to a specific theme, including both relevance and
    sentiment dimensions with detailed reasoning.
    
    Attributes:
        reason: Detailed explanation of scoring rationale, referencing
                analyzed quotes and evaluation criteria
        relevance_score: Company's relevance to the theme:
                        0 = Not relevant to the theme
                        1 = Slightly relevant  
                        2 = Highly relevant
        sentiment_score: Company's sentiment regarding the theme:
                        -1 = Negative sentiment
                         0 = Neutral sentiment
                         1 = Positive sentiment
    """
    reason: str
    relevance_score: int
    sentiment_score: int
