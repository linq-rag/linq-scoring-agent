from typing import List, Dict

_SYSTEM_OVERALL_FILTER_PROMPT = """
<task>
You are a financial analyst filtering key investment insights from earnings call transcripts.  
Retain only phrases that provide clear, actionable insights or reveal meaningful speaker sentiment.
</task>

<procedure>
1. **Select Key Sentences**  
   - Include only phrases impacting investment decisions (financials, strategy, risks, market position).  
   - Remove vague, promotional, repetitive, or non-actionable phrases.
   - Evaluate objectively and apply a uniform standard across transcripts.
2. **Assign Sentiment**
   - **1 (Positive):** Indicates strengths, growth, or opportunities.  
   - **0 (Neutral):** Factual with no direct investment impact.
   - **-1 (Negative):** Highlights risks, weaknesses, or challenges.  
</procedure>

<format>
### Output Format:
Provide results in the following JSON format:
```json
    "related_indices": [index1, index2, ...],
    "sentiment_scores": [score1, score2, score3, ...]
```
</format>  
"""

_USER_OVERALL_FILTER_PROMPT = """
<company>
{company_name}  
</company>

<instruction>
Analyze the provided phrases and determine whether they offer meaningful investment insights or reveal critical speaker sentiment.
</instruction>

<phrases>
{quotes}
</phrases>
"""


def get_overall_filtering_messages(company_name: str, quotes: str) -> List[Dict[str, str]]:

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_OVERALL_FILTER_PROMPT
        },
        {
            "role": "user",
            "content": _USER_OVERALL_FILTER_PROMPT.format(company_name=company_name, quotes=quotes)
        },
    ]
    return messages
