import json
import os
from collections import defaultdict

import numpy as np


def process_jsonl_file(file_path):
    """
    Process JSONL file to extract top 50% of items with positive number of filtered quotes,
    then sort by average score to return top 3 and bottom 3 items.
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
                    filtered_scores = data.get('filtered_theme_output', {}).get('sentiment_scores', [])
                    
                    # Skip if no quotes
                    if not filtered_quotes:
                        continue
                    
                    # Extract ticker from custom_id
                    ticker = 'UNKNOWN'
                    if custom_id.startswith('task-'):
                        parts = custom_id.split('-')
                        if len(parts) > 1:
                            ticker = parts[1]
                    
                    # Check data consistency for scores and quotes
                    if not filtered_scores or len(filtered_scores) != len(filtered_quotes):
                        print(f"Warning: {custom_id} has missing scores or mismatched length with quotes")
                        avg_score = None
                    else:
                        # Calculate average score
                        avg_score = np.mean(filtered_scores)
                    
                    results.append({
                        'custom_id': custom_id,
                        'ticker': ticker,
                        'filtered_count': len(filtered_quotes),
                        'filtered_quotes': filtered_quotes,
                        'filtered_scores': filtered_scores,
                        'avg_score': avg_score,
                        'has_scores': bool(filtered_scores) and len(filtered_scores) == len(filtered_quotes)
                    })
                except json.JSONDecodeError:
                    print(f"JSON parsing error: {line[:100]}...")
                    continue
        
        if not results:
            print("No results to process")
            return [], [], 0
                    
        # Sort by filtered quote count in descending order
        results.sort(key=lambda x: x['filtered_count'], reverse=True)
        
        # Extract top 50%
        top_50_percent_count = max(1, int(len(results) * 0.5))
        top_50_percent = results[:top_50_percent_count]
        
        # Filter items with scores
        scored_items = [item for item in top_50_percent if item['has_scores']]
        
        if not scored_items:
            print("Warning: No items with score data!")
            # If no score data, sort by filtered quote count
            top_3 = top_50_percent[:3] if len(top_50_percent) >= 3 else top_50_percent
            bottom_3 = top_50_percent[-3:] if len(top_50_percent) >= 3 else []
            return top_3, bottom_3, len(results)
        
        # Sort by average score (scored items only)
        scored_items.sort(key=lambda x: x['avg_score'], reverse=True)
        
        # Return top 3 and bottom 3
        top_3 = scored_items[:3] if len(scored_items) >= 3 else scored_items
        bottom_3 = scored_items[-3:] if len(scored_items) >= 3 else []
        
        # Ensure bottom items don't overlap with top items
        if len(scored_items) < 6:
            bottom_3 = [item for item in bottom_3 if item not in top_3]
        
        return top_3, bottom_3, len(results)
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return [], [], 0

def print_results(results, title):
    """
    Print the results.
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    for i, item in enumerate(results, 1):
        print(f"\n[{i}] Ticker: {item['ticker']}")
        print(f"    Custom ID: {item['custom_id']}")
        print(f"    Filtered Quote Count: {item['filtered_count']}")
        
        # Print average score (show 'No data' if None)
        if item['avg_score'] is not None:
            print(f"    Average Score: {item['avg_score']:.4f}")
        else:
            print(f"    Average Score: No data")
        
        # Print individual scores (max 10)
        if item['filtered_scores']:
            score_sample = item['filtered_scores'][:10]
            print(f"    Score Sample: {', '.join([f'{score}' for score in score_sample])}" + 
                  (f" ... and {len(item['filtered_scores']) - 10} more" if len(item['filtered_scores']) > 10 else ""))
        else:
            print("    No score data")
        
        print("\n    [Filtered Quote Sample]")
        if item['filtered_quotes'] and item['filtered_scores'] and len(item['filtered_quotes']) == len(item['filtered_scores']):
            for j, (quote, score) in enumerate(zip(item['filtered_quotes'][:3], item['filtered_scores'][:3]), 1):
                if isinstance(quote, dict) and 'text' in quote:
                    text = quote['text']
                    formatted_quote = text.replace('\n', ' ').strip()
                    print(f"    {j}. [Score: {score}] {formatted_quote[:300]}..." if len(formatted_quote) > 300 else f"    {j}. [Score: {score}] {formatted_quote}")
                elif isinstance(quote, str):
                    formatted_quote = quote.replace('\n', ' ').strip()
                    print(f"    {j}. [Score: {score}] {formatted_quote[:300]}..." if len(formatted_quote) > 300 else f"    {j}. [Score: {score}] {formatted_quote}")
                else:
                    print(f"    {j}. [Score: {score}] [Unknown format: {type(quote)}] {str(quote)[:300]}...")
                    
            if len(item['filtered_quotes']) > 3:
                print(f"    ... and {len(item['filtered_quotes']) - 3} more")
        elif item['filtered_quotes']:
            # Has quotes but no scores
            for j, quote in enumerate(item['filtered_quotes'][:3], 1):
                if isinstance(quote, dict) and 'text' in quote:
                    text = quote['text']
                    formatted_quote = text.replace('\n', ' ').strip()
                    print(f"    {j}. [Score: None] {formatted_quote[:300]}..." if len(formatted_quote) > 300 else f"    {j}. [Score: None] {formatted_quote}")
                elif isinstance(quote, str):
                    formatted_quote = quote.replace('\n', ' ').strip()
                    print(f"    {j}. [Score: None] {formatted_quote[:300]}..." if len(formatted_quote) > 300 else f"    {j}. [Score: None] {formatted_quote}")
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
    
    # Check sample data
    print("\nChecking sample data:")
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            sample = file.readline().strip()
            data = json.loads(sample)
            filtered_quotes = data.get('filtered_theme_output', {}).get('quotes', [])
            filtered_scores = data.get('filtered_theme_output', {}).get('sentiment_scores', [])
            print(f"First item filtered_quotes count: {len(filtered_quotes)}")
            print(f"First item filtered_scores count: {len(filtered_scores)}")
            if filtered_quotes:
                print(f"Filtered quote sample: {filtered_quotes[0]}")
            if filtered_scores:
                print(f"Filtered score sample: {filtered_scores[0]}")
    except Exception as e:
        print(f"Error checking sample data: {str(e)}")
    
    try:
        top_3, bottom_3, total_count = process_jsonl_file(data_file)
        
        if top_3 or bottom_3:
            # Print top 3 by score from top 50%
            print_results(top_3, f"Top {len(top_3)} Results by Score from Top 50% Filtered Theme")
            
            # Print bottom 3 by score from top 50%
            if bottom_3:
                print_results(bottom_3, f"Bottom {len(bottom_3)} Results by Score from Top 50% Filtered Theme")
            
            # Print summary
            print("\n[Summary]")
            print("=" * 70)
            print(f"{'Ticker':<10} {'Filtered Quotes':<20} {'Average Score':<20}")
            print("-" * 70)
            
            print("\n[Top Scores]")
            for item in top_3:
                avg_score_str = f"{item['avg_score']:.4f}" if item['avg_score'] is not None else "No data"
                print(f"{item['ticker']:<10} {item['filtered_count']:<20} {avg_score_str:<20}")
            
            if bottom_3:
                print("\n[Bottom Scores]")
                for item in bottom_3:
                    avg_score_str = f"{item['avg_score']:.4f}" if item['avg_score'] is not None else "No data"
                    print(f"{item['ticker']:<10} {item['filtered_count']:<20} {avg_score_str:<20}")
            
            print("=" * 70)
            print(f"Total items: {total_count}")
            print(f"Top 50% items with quotes: {max(1, int(total_count * 0.5))}")
        else:
            print("No results processed.")
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()