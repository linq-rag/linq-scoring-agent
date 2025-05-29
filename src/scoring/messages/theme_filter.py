"""
Theme Filtering Prompt Messages

This module provides prompt templates for filtering theme-specific quotes
for relevance and assigning sentiment scores. It implements strict filtering
criteria to ensure only highly relevant content is retained.
"""

from typing import Dict, List

_SYSTEM_THEME_FILTER_PROMPT = """
<task>
You are an expert assistant filtering text strictly relevant to a given theme.  
</task>

<guidelines>  
1. **Identify Relevant Sentences**  
   - Retain only phrases that fully and completely express the theme's core meaning.  
   - Remove generic, vague, speculative, or loosely related statements.
   - Omit content that introduces unrelated elements.
   - Apply strict, factual criteria without interpretation.
   - Keep only phrases that explicitly reinforce the theme.
2. **Assign Sentiment Scores**
   - **1 (Positive):** Highlights strengths, opportunities, or growth potential.  
   - **0 (Neutral):** Provides factual information with no clear impact.
   - **-1 (Negative):** Indicates risks, weaknesses, or challenges.
</guidelines>

<format>
```json
    "included_phrases_indices": [1, 2, 5],
    "sentiment_scores": [score1, score2, score3, ...]
```
</format>
"""

_USER_THEME_FILTER_PROMPT = """
<company>
{company_name}
</company>

<task>
Evaluate the provided phrases for relevance to the theme:
**{theme}**
</task>

<quotes>
{quotes}
</quotes>
"""


def get_theme_filtering_messages(company_name: str, theme: str, quotes: str) -> List[Dict[str, str]]:
    """
    Generate LLM messages for theme-specific quote filtering and sentiment analysis.
    
    This function creates prompts for filtering previously extracted quotes to retain
    only those that are strictly relevant to the specified theme. It also assigns
    sentiment scores to each retained quote.
    
    Args:
        company_name: Name of the company for contextual analysis
        theme: Specific theme description for filtering relevance
        quotes: Formatted quotes text to be filtered
        
    Returns:
        List of message dictionaries with system and user prompts
        for theme-specific filtering and sentiment scoring
        
    Note:
        The filtering criteria are intentionally strict to ensure high-quality
        thematic alignment and avoid loosely related content
    """
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_THEME_FILTER_PROMPT
        },
        {
            "role": "user",
            "content": _USER_THEME_FILTER_PROMPT.format(company_name=company_name, theme=theme, quotes=quotes)
        },
    ]
    return messages
