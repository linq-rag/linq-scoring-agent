import json
from typing import Dict, List

from src.scoring.outputs import ExtractedOutput

_SYSTEM_THEME_EXTRACT_PROMPT = f"""
<task>
You are an investing expert analyzing earnings call transcripts for insights on a given theme.  
Extract only sentences that fully and exclusively align with the theme.
</task>

<procedure>
1. **Identify Relevant Sentences**
   - Select sentence indices that exactly match the theme’s meaning, with no added or missing elements.  
   - The sentence must fully express the theme.
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

_USER_THEME_EXTRACT_PROMPT = """
<company>
{company_name}
</company>

<task>
1. Analyze the transcript for the theme: 

```{theme}```

2. Select related sentence indices and assign scores.
3. Ensure selected sentences clearly align with the theme.
</task>

<transcript>
{transcript}
</transcript>
"""


def get_theme_extracting_messages(company_name: str, theme: str, transcript: str) -> List[Dict[str, str]]:

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_THEME_EXTRACT_PROMPT
        },
        {
            "role": "user",
            "content": _USER_THEME_EXTRACT_PROMPT.format(company_name=company_name, theme=theme, transcript=transcript)
        },
    ]
    return messages
