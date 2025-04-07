import json
import os
from collections import defaultdict


def process_jsonl_file(file_path):
    """
    Process JSONL file and return top 5 items based on filtered theme output.
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract necessary data
                    custom_id = data.get('custom_id', '')
                    filtered_quotes = data.get('filtered_theme_output', {}).get('quotes', [])
                    extracted_quotes = data.get('extracted_theme_output', {}).get('quotes', [])
                    
                    # Extract ticker from custom_id
                    # Example: task-DLR-22-02-17-21_4Q_THEME -> DLR
                    ticker = 'UNKNOWN'
                    if custom_id.startswith('task-'):
                        parts = custom_id.split('-')
                        if len(parts) > 1:
                            ticker = parts[1]  # Second part is likely the ticker
                    
                    results.append({
                        'custom_id': custom_id,
                        'ticker': ticker,
                        'filtered_count': len(filtered_quotes),
                        'extracted_count': len(extracted_quotes),
                        'filtered_quotes': filtered_quotes,
                        'extracted_quotes': extracted_quotes
                    })
                except json.JSONDecodeError:
                    print(f"JSON parsing error: {line[:100]}...")
                    continue
                    
        # Sort by filtered_count in descending order
        results.sort(key=lambda x: x['filtered_count'], reverse=True)
        
        return results[:5]  # Return top 5
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def print_top_results(results):
    """
    Print top results.
    """
    print("\n" + "=" * 80)
    print(f"{'Top 5 Results by Filtered Theme Output':^80}")
    print("=" * 80)
    
    for i, item in enumerate(results, 1):
        print(f"\n[{i}] Ticker: {item['ticker']}")
        print(f"    Custom ID: {item['custom_id']}")
        print(f"    Filtered Quote Count: {item['filtered_count']} | Extracted Quote Count: {item['extracted_count']}")
        
        print("\n    [Filtered Quote Sample]")
        if item['filtered_quotes']:
            for j, quote in enumerate(item['filtered_quotes'][:3], 1):  # Print first 3 only
                formatted_quote = quote.replace('\n', ' ').strip()
                print(f"    {j}. {formatted_quote[:300]}..." if len(formatted_quote) > 300 else f"    {j}. {formatted_quote}")
            if len(item['filtered_quotes']) > 3:
                print(f"    ... and {len(item['filtered_quotes']) - 3} more")
        else:
            print("    (None)")
        
        print("\n" + "-" * 80)

def main():
    # Set data file path
    data_file = "/Users/junekwon/Desktop/Projects/scoring_agent/data/final/data/2021_4Q/21_4q_theme_ai_(artificial_intelligence).jsonl"
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: File not found - {data_file}")
        return
    
    print(f"Processing file: {data_file}")
    results = process_jsonl_file(data_file)
    
    if results:
        print_top_results(results)
        
        # Print summary
        print("\n[Summary]")
        print("=" * 80)
        print(f"{'Ticker':<10} {'Filtered Quotes':<20} {'Extracted Quotes':<20}")
        print("-" * 80)
        for item in results:
            print(f"{item['ticker']:<10} {item['filtered_count']:<20} {item['extracted_count']:<20}")
        print("=" * 80)
    else:
        print("No results processed.")

if __name__ == "__main__":
    main() 