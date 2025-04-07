#!/usr/bin/env python3
import json
import sys


def main():
    """
    JSONL 파일에서 custom_id와 filtered_theme_output 필드를 추출하여 10개 출력합니다.
    """
    if len(sys.argv) < 2:
        print("사용법: python print_jsonl_simple.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if count >= 10:  # 10개 레코드만 출력
                    break
                
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id', 'N/A')
                    filtered_output = data.get('filtered_theme_output', {})
                    
                    print(f"===== {count+1}번째 레코드 =====")
                    print(f"Custom ID: {custom_id}")
                    
                    # filtered_theme_output 내용 처리
                    quotes = filtered_output.get('quotes', [])
                    sentiment_scores = filtered_output.get('sentiment_scores', [])
                    
                    print("Filtered Theme Output:")
                    print(f"  - 인용구 수: {len(quotes)}")
                    if quotes:
                        print("  - 인용구 예시:")
                        for i, quote in enumerate(quotes[:2], 1):  # 처음 2개만 출력
                            print(f"    {i}. {quote[:100]}..." if len(quote) > 100 else f"    {i}. {quote}")
                    
                    print(f"  - 감정 점수: {sentiment_scores}")
                    print("-" * 50)
                    
                    count += 1
                except json.JSONDecodeError:
                    print(f"잘못된 JSON 형식: {line[:100]}...")
                    continue
                    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 