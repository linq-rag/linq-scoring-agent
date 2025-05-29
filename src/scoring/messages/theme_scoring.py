"""
Theme Scoring Prompt Messages

This module provides prompt templates for scoring companies' relevance and
sentiment regarding specific themes based on filtered quotes from earnings calls.
It implements a dual-scoring system for both thematic relevance and sentiment.
"""

from typing import Dict, List

_SYSTEM_SCORE_PROMPT = """
<task>
You are an expert assistant analyzing a company's earnings call transcript for alignment with the theme.

1. **Theme Relevance:**  
   - Assign a score:  
     - **0:** Not relevant  
     - **1:** Moderately relevant  
     - **2:** Highly relevant  
   - If no quotes are provided, return **0**.  

2. **Sentiment Analysis:**  
   - Assign a score:  
     - **-1:** Negative (stock decline expected)  
     - **0:** Neutral (stock stability expected)  
     - **1:** Positive (stock growth expected)  
   - If no quotes are provided, return **0**.  

3. **Context Considerations:**  
   - Assess industry, competition, revenue impact, growth potential, and risks.  

4. **Reasoning Structure:**  
   - **Overview:** Summary of alignment with the theme  
   - **Analysis:** Revenue impact, market trends, growth drivers, strategic risks  
   - **Conclusion:** Justify scores with supporting quotes  
</task>

<format>
```json
{
  "reason": "<detailed_reasoning>",
  "relevance_score": <0-2>,
  "sentiment_score": <−1 to 1>
}
```
</format>
"""

_USER_SCORE_PROMPT = """
<company>
{company_name}
</company>

<instruction>
Analyze the provided theme-related quotes for relevance and sentiment.
</instruction>

<quotes>
{quotes}
</quotes>
"""


def get_theme_scoring_messages(company_name: str, quotes: str) -> List[Dict[str, str]]:
    """
    Generate LLM messages for theme-specific relevance and sentiment scoring.
    
    This function creates prompts for scoring a company's relationship to a specific
    theme based on filtered quotes from earnings calls. It implements a dual-scoring
    approach that evaluates both thematic relevance (0-2) and sentiment (-1 to 1).
    
    Args:
        company_name: Name of the company being analyzed
        quotes: Formatted theme-relevant quotes for scoring
        
    Returns:
        List of message dictionaries with system and user prompts
        for comprehensive theme scoring analysis
        
    Note:
        The scoring considers industry context, competitive positioning,
        revenue impact, and strategic implications for comprehensive evaluation
    """
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_SCORE_PROMPT
        },
        {
            "role": "user",
            "content": _USER_SCORE_PROMPT.format(company_name=company_name, quotes=quotes)
        },
    ]
    return messages
