#!/usr/bin/env python3
import json
import pprint
import sys


def print_jsonl_fields(file_path, field_names, limit):
    """
    JSONL 파일에서 지정된 필드만 추출하여 출력합니다.
    
    Args:
        file_path (str): JSONL 파일 경로
        field_names (list): 추출할 필드 이름 목록
        limit (int): 출력할 레코드 수
    """
    count = 0
    pp = pprint.PrettyPrinter(indent=2, width=100)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if count >= limit:
                    break
                    
                try:
                    data = json.loads(line)
                    print(f"===== 레코드 {count + 1} =====")
                    
                    for field in field_names:
                        if field in data:
                            print(f"[{field}]")
                            if isinstance(data[field], (dict, list)):
                                pp.pprint(data[field])
                            else:
                                print(data[field])
                            print()
                    
                    print("-" * 50)
                    count += 1
                except json.JSONDecodeError:
                    print(f"JSON 파싱 오류: {line}")
                    continue
    except Exception as e:
        print(f"파일 읽기 오류: {e}")

if __name__ == "__main__":
        
    file_path = "/Users/junekwon/Desktop/Projects/scoring_agent/data/4o-mini/2023_2Q-groq/23_2q_theme_interest_rate_hikes.jsonl"
    limit = 10
    
    # 추출할 필드 이름
    fields_to_extract = ["custom_id", "filtered_theme_output"]
    
    print_jsonl_fields(file_path, fields_to_extract, limit) 