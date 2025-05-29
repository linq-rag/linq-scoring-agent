"""
Overall Sentiment Extraction Prompt Messages

This module provides prompt templates for extracting financially relevant
content from earnings call transcripts for overall sentiment analysis.
It focuses on broad investment insights rather than specific themes.
"""

import json
from typing import Dict, List

from src.scoring.outputs import ExtractedOutput

_SYSTEM_OVERALL_EXTRACT_PROMPT = f"""
<task>
You are a financial analyst reviewing an Earnings Call Transcript.
Extract key investment insights by identifying sentences with financial relevance and sentiment impact.
</task>

<procedure>
1. **Select Key Sentences**
   - Focus on company performance, growth, risks, and financial outlook.  
   - Exclude generic or non-investment statements.
   - If no sentences match, return empty lists.
</procedure>

<format>
Return the extracted data in the following format: 
- Structure your output in the following JSON format:  
```json
    "indices": [index1, index2, index3, ...],
```
If no indices match the criteria, return an empty list while maintaining the following format.
```json
    "indices": [],
```
</format>
"""

_USER_OVERALL_EXTRACT_PROMPT = """
Select indices of key sentences.
<company>
{company_name}
</company>

<transcript>
{transcript}
</transcript>
"""


def get_overall_extracting_messages(company_name: str, transcript: str) -> List[Dict[str, str]]:
    """
    Generate LLM messages for overall financial sentiment extraction.
    
    This function creates prompts for extracting broadly relevant financial
    content from earnings call transcripts. Unlike theme-specific extraction,
    this focuses on general investment insights and company performance indicators.
    
    Args:
        company_name: Name of the company for contextual analysis
        transcript: Formatted transcript text with numbered quotes
        
    Returns:
        List of message dictionaries with system and user prompts
        for overall sentiment analysis
        
    Note:
        The extraction criteria prioritize financial relevance and
        sentiment impact over specific thematic alignment
    """
    messages = [
        {
            "role": "system", 
            "content": _SYSTEM_OVERALL_EXTRACT_PROMPT
        },
        {
            "role": "user",
            "content": _USER_OVERALL_EXTRACT_PROMPT.format(company_name=company_name, transcript=transcript)
        },
    ]
    return messages
