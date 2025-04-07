import json
import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a `.env` file
load_dotenv()

# Initialize the OpenAI client with API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API"))

MONTH_MAPPING = {
    "4": "January",
    "1": "April",
    "2": "July",
    "3": "October",
}


def extract_wiki(url, start_section, end_sections):

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()

        # 특정 섹션 추출
        start_index = page_text.find(start_section)

        end_index = -1
        for end_section in end_sections:
            end_index = max([end_index, page_text.find(end_section)])

        if start_index != -1 and end_index != -1:
            extracted_text = page_text[start_index + len(start_section):end_index]

            # 불필요한 공백 제거 및 줄바꿈 정리
            cleaned_text = '\n'.join([line.strip() for line in extracted_text.splitlines() if line.strip()])

            return cleaned_text

        else:
            print("Could not find the specified sections in the text.")
            return None
    return None


def extract_wiki_with_year_quarter(year: str, quarter: str):
    if not quarter == "4":
        start_year = str(int(year) - 1)
        end_year = str(int(year))
    else:
        start_year = str(int(year))
        end_year = str(int(year) + 1)

    quarter_month = MONTH_MAPPING[quarter]

    text = extract_wiki(
        url=f"https://en.wikipedia.org/wiki/{start_year}",
        start_section=f"{quarter_month}[edit]" if end_year < "2024" else f"{quarter_month}",
        end_sections=["Births and deaths[edit]", "Demographics[edit]"] if end_year < "2024" else ["Deaths"]
    )
    print(text)
    text += "\n"

    text += extract_wiki(
        url=f"https://en.wikipedia.org/wiki/{end_year}",
        start_section=f"January[edit]" if end_year < "2024" else "January",
        end_sections=[f"{quarter_month}[edit]"] if end_year < "2024" else [f"{quarter_month}"]
    )
    print(text)
    return text


SYSTEM = """
<task>
I will provide a timeline of events covering a one-year period.
Based on this data and the themes identified in the previous quarter, determine 20 investment-relevant themes having 1~2 keyword, for the current quarter.
</task>

<instruction>
- The output must be structured as a valid JSON object.
- Extract and list **20 investable themes that can be directly used for portfolio allocation**, strictly based on the provided Wikipedia text.
- Themes must be directly linked to **investment opportunities**, including sectors, asset classes, technologies, or financial instruments.
- **Avoid general political events, policy changes, or economic conditions unless they clearly translate into an investable theme.**

<validation>
- Ensure that each theme is explicitly supported by a verbatim quote from Wikipedia.
- Reject any themes that do not clearly indicate a **direct investment opportunity or portfolio-level investable keyword**.
</validation>

<format>
The JSON output must follow this format:
The key should be a **well-formed actionable investment theme name** and the value should be a **related verbatim quote** from Wikipedia **exactly as written**.
```json
{
    "theme_1": "Exact verbatim quote related to `theme_1` from Wikipedia",
    "theme_2": "Exact verbatim quote related to `theme_2` from Wikipedia",
    ...
}
```
</format>
"""

USER = """
<date>
{date}
</date>

<wikipedia>
{text}
</wikipedia>

<instruction>
Analyze the key **investment-focused** themes that have significantly influenced global markets during the given period.
Identify 20 **investable** themes having high-level 1~2 keyword, strictly based on the provided Wikipedia text.
- Themes must be **directly applicable to investment decisions**.
- **Avoid general macroeconomic terms or political events unless they have clear investment implications.**
- Provide the response strictly in JSON format.
</instruction>
"""


def get_theme_list(
        wikipedia_text: str,
        date: str,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
):
    user_prompt_with_formatting = USER.format(text=wikipedia_text, date=date)

    # Simulate OpenAI API client call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt_with_formatting},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    message = json.loads(response.choices[0].message.content)

    return message


if __name__ == "__main__":
    _year = "2023"
    _quarter = "3"
    _date = _year + "-" + MONTH_MAPPING[_quarter]

    data = get_theme_list(wikipedia_text=extract_wiki_with_year_quarter(_year, _quarter), date=_date)

    print(data)
    print(list(data.keys()))
