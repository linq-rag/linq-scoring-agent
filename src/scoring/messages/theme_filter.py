from typing import List, Dict

_SYSTEM_THEME_FILTER_PROMPT = """
<task>
You are an expert assistant filtering text strictly relevant to a given theme.  
</task>

<guidelines>  
1. **Identify Relevant Sentences**  
   - Retain only phrases that fully and completely express the theme’s core meaning.  
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
