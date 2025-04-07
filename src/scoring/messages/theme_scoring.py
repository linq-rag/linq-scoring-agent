from typing import List, Dict

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
