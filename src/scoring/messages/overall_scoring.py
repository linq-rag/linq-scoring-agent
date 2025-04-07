from typing import List, Dict

_SYSTEM_OVERALL_SCORE_PROMPT = """
<task>
You are a financial analyst evaluating an earnings call transcript to assess sentiment and justify your analysis.
</task>

<criteria>
1. Predict sentiment on a scale from -1 to 1 based on its expected impact on stock performance:
   - -1: Negative (Stock decline expected)
   - 0: Neutral (Stock stability expected)
   - 1: Positive (Stock growth expected)

2. Provide a justification that explains the assigned sentiment.
   - Consider industry trends, competitors, and market conditions.
   - Support your assessment with referring given quotes.
   - Balance risks and opportunities in your analysis.
</criteria>

<format>
### Output Format:
```json
{
  "reason": "<detailed_reasoning>",
  "score": <final_score>
}
```
</format>
"""

_USER_OVERALL_SCORE_PROMPT = """
<company>
company_name}
</company>

<instruction>
Analyze the provided quotes and assess the company's sentiment.
</instruction>

<quotes>
{quotes}
</quotes>
"""


def get_overall_scoring_messages(company_name: str, quotes: str) -> List[Dict[str, str]]:

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_OVERALL_SCORE_PROMPT
        },
        {
            "role": "user",
            "content": _SYSTEM_OVERALL_SCORE_PROMPT.format(company_name=company_name, quotes=quotes)
        },
    ]
    return messages
