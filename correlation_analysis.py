import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def load_stock_price_data(stock_price_file):
    """
    Load stock price data file and organize data by ticker.
    """
    stock_price_dict = {}
    
    try:
        with open(stock_price_file, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    ticker = data.get('ticker')
                    
                    if not ticker:
                        continue
                    
                    stock_prices = data.get('stock_prices', [])
                    
                    # # 날짜 기준으로 정렬 (최신 날짜 순)
                    # stock_prices.sort(key=lambda x: x.get('date', ''), reverse=True)
                    
                    stock_price_dict[ticker] = {
                        'stock_prices': stock_prices
                    }
                    
                except json.JSONDecodeError:
                    continue
        
        print(f"Stock price data loaded: {len(stock_price_dict)} tickers loaded")
        return stock_price_dict
    
    except Exception as e:
        print(f"Error processing stock price data file: {str(e)}")
        return {}


def load_and_analyze_theme_data(theme_file, stock_price_dict, is_overall=False):
    """
    Load theme data file and calculate CAR(0,1) using stock price data.
    """
    results = []
    
    try:
        with open(theme_file, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract necessary data
                    custom_id = data.get('custom_id', '')
                    
                    # Use appropriate output field based on theme type
                    output_field = 'filtered_overall_output' if is_overall else 'filtered_theme_output'
                    filtered_quotes = data.get(output_field, {}).get('quotes', [])
                    filtered_scores = data.get(output_field, {}).get('sentiment_scores', [])
                    
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
                        continue
                    
                    # Skip if no stock price data for ticker
                    if ticker not in stock_price_dict:
                        print(f"Warning: No stock price data for ticker {ticker}")
                        continue
                    
                    # Identify event date
                    event_date = None
                    if custom_id.startswith('task-'):
                        parts = custom_id.split('-')
                        if len(parts) >= 4:  # task-TICKER-YY-MM-DD format
                            year = '20' + parts[2]  # Convert YY to YYYY
                            month = parts[3]
                            day = parts[4].split('_')[0]  # Extract DD from DD_remainder format
                            event_date = f"{year}-{month}-{day}"
                    
                    if not event_date:
                        print(f"Warning: Cannot extract event date from {custom_id}")
                        continue
                    
                    # Find event date and next day in stock price data
                    stock_prices = stock_price_dict[ticker]['stock_prices']
                    
                    event_day_data = None
                    next_day_data = None
                    
                    # Find event date
                    for i, price_data in enumerate(stock_prices):
                        if price_data.get('date') == event_date:
                            event_day_data = price_data
                            # Check for next day data (dates might be reverse sorted)
                            for j, next_price_data in enumerate(stock_prices):
                                if j != i:  # Must not be same data
                                    next_date = next_price_data.get('date', '')
                                    # Check if next trading day
                                    if is_next_trading_day(event_date, next_date):
                                        next_day_data = next_price_data
                                        break
                            break
                    
                    # If event date or next day data not found
                    if not event_day_data or not next_day_data:
                        # Alternative: find consecutive days with abnormal returns
                        for i in range(len(stock_prices) - 1):
                            current_data = stock_prices[i]
                            next_data = stock_prices[i+1]
                            
                            event_day_ar = current_data.get('abnormal_return')
                            next_day_ar = next_data.get('abnormal_return')
                            
                            if event_day_ar is not None and next_day_ar is not None:
                                event_day_data = current_data
                                next_day_data = next_data
                                break
                    
                    # Skip if still no appropriate data found
                    if not event_day_data or not next_day_data:
                        print(f"Warning: No suitable event date or consecutive abnormal return data found for {ticker}({event_date})")
                        continue
                    
                    # Extract abnormal return values
                    event_day_ar = event_day_data.get('abnormal_return')
                    next_day_ar = next_day_data.get('abnormal_return')
                    
                    # Check if all abnormal return values exist
                    if event_day_ar is None or next_day_ar is None:
                        print(f"Warning: Missing abnormal return values for {ticker}({event_date})")
                        continue
                    
                    # Calculate CAR(0, 1)
                    car_m1_p1 = (1 + event_day_ar) * (1 + next_day_ar) - 1
                    
                    # Store data
                    avg_sentiment_score = np.mean(filtered_scores)
                    
                    results.append({
                        'ticker': ticker,
                        'custom_id': custom_id,
                        'event_date': event_date,
                        'filtered_count': len(filtered_quotes),
                        'avg_sentiment_score': avg_sentiment_score,
                        'car_m1_p1': car_m1_p1,
                        'event_day_ar': event_day_ar,
                        'next_day_ar': next_day_ar
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Error processing data: {str(e)}")
                    continue
        
    except Exception as e:
        print(f"Error processing theme data file: {str(e)}")
    
    if not results:
        print("No results to process")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def is_next_trading_day(date1, date2):
    """
    Check if date2 is the next trading day after date1.
    Trading day gap can be 1-3 days (considering weekends and holidays).
    """
    try:
        from datetime import datetime

        # 날짜 문자열을 datetime 객체로 변환
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        
        # date2가 date1보다 이후 날짜인지 확인
        if d2 > d1:
            # 두 날짜 간의 일수 차이 계산
            delta = (d2 - d1).days
            # 일반적인 거래일 간격은 1-3일 (주말, 공휴일 고려)
            return 1 <= delta <= 3
        return False
    except Exception:
        # 날짜 형식 오류 등이 발생하면 False 반환
        return False


def filter_top_data_by_quotes(df, percentage=0.5):
    """
    Extract top percentage% of data based on number of filtered quotes.
    """
    if df is None or len(df) == 0:
        return None
    
    # Sort by filtered quote count in descending order
    df_sorted = df.sort_values('filtered_count', ascending=False)
    
    # Extract top percentage%
    top_count = max(1, int(len(df_sorted) * percentage))
    df_top = df_sorted.head(top_count)
    
    print(f"Total data count: {len(df)}")
    print(f"Top {percentage*100}% data count by quotes: {len(df_top)}")
    
    return df_top


def analyze_correlation(df):
    """
    Analyze correlation between average sentiment score and CAR(0,1).
    """
    if df is None or len(df) < 2:
        print("Insufficient data for correlation analysis.")
        return None, None, None
    
    # Calculate correlation
    correlation, p_value = pearsonr(df['avg_sentiment_score'], df['car_m1_p1'])
    
    print(f"\nCorrelation Analysis Results:")
    print(f"Number of data points: {len(df)}")
    print(f"Pearson correlation coefficient between average sentiment score and CAR(0,1): {correlation:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        significance = "Statistically significant (p < 0.05)"
    else:
        significance = "Not statistically significant (p >= 0.05)"
    
    print(f"Significance: {significance}")
    
    return correlation, p_value, significance


def plot_correlation(df, correlation, p_value, significance, output_path=None):
    """
    Visualize the correlation.
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df['avg_sentiment_score'], df['car_m1_p1'], alpha=0.6)
    
    # Add trend line
    m, b = np.polyfit(df['avg_sentiment_score'], df['car_m1_p1'], 1)
    plt.plot(df['avg_sentiment_score'], m * df['avg_sentiment_score'] + b, color='red')
    
    plt.title(f'Correlation between Average Sentiment Score and CAR(0,1)\nCorrelation: {correlation:.4f}, {significance}')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('CAR(0,1)')
    plt.grid(True, alpha=0.3)
    
    # Add correlation text
    text = f"Correlation: {correlation:.4f}\np-value: {p_value:.4f}"
    plt.annotate(
        text, xy=(0.05, 0.95), xycoords='axes fraction', 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Graph saved to {output_path}")
    
    plt.show()


def generate_summary_table(df):
    """
    Generate a summary table of results.
    """
    print("\n[Summary Table]")
    print("=" * 90)
    print(f"{'Ticker':<8} {'Date':<12} {'Filtered Quotes':<18} {'Avg Sentiment':<16} {'CAR(0,1)':<12} {'AR(0)':<10} {'AR(1)':<10}")
    print("-" * 90)
    
    # Top 5 by average sentiment score
    df_top = df.sort_values('avg_sentiment_score', ascending=False).head(5)
    print("\n[Top 5 by Average Sentiment Score]")
    for _, row in df_top.iterrows():
        print(f"{row['ticker']:<8} {row['event_date']:<12} {row['filtered_count']:<18} {row['avg_sentiment_score']:.4f}{' '*11} {row['car_m1_p1']:.4f}{' '*7} {row['event_day_ar']:.4f}{' '*5} {row['next_day_ar']:.4f}")
    
    # Bottom 5 by average sentiment score
    df_bottom = df.sort_values('avg_sentiment_score').head(5)
    print("\n[Bottom 5 by Average Sentiment Score]")
    for _, row in df_bottom.iterrows():
        print(f"{row['ticker']:<8} {row['event_date']:<12} {row['filtered_count']:<18} {row['avg_sentiment_score']:.4f}{' '*11} {row['car_m1_p1']:.4f}{' '*7} {row['event_day_ar']:.4f}{' '*5} {row['next_day_ar']:.4f}")
    
    # Top 5 by CAR(0,1)
    df_car_top = df.sort_values('car_m1_p1', ascending=False).head(5)
    print("\n[Top 5 by CAR(0,1)]")
    for _, row in df_car_top.iterrows():
        print(f"{row['ticker']:<8} {row['event_date']:<12} {row['filtered_count']:<18} {row['avg_sentiment_score']:.4f}{' '*11} {row['car_m1_p1']:.4f}{' '*7} {row['event_day_ar']:.4f}{' '*5} {row['next_day_ar']:.4f}")
    
    # Bottom 5 by CAR(0,1)
    df_car_bottom = df.sort_values('car_m1_p1').head(5)
    print("\n[Bottom 5 by CAR(0,1)]")
    for _, row in df_car_bottom.iterrows():
        print(f"{row['ticker']:<8} {row['event_date']:<12} {row['filtered_count']:<18} {row['avg_sentiment_score']:.4f}{' '*11} {row['car_m1_p1']:.4f}{' '*7} {row['event_day_ar']:.4f}{' '*5} {row['next_day_ar']:.4f}")
    
    print("=" * 90)


def process_quarter_data(data_dir, output_base_dir):
    """
    Process all theme files in a quarter directory and generate correlation analysis results.
    
    Args:
        data_dir: Directory containing theme and stock price files (e.g., '.../2021_4Q')
        output_base_dir: Base directory for saving correlation results (e.g., '.../figures/Corr')
    """
    # Extract quarter info from directory name
    quarter = os.path.basename(data_dir)  # e.g., '2021_4Q'
    
    # Find stock prices file
    stock_price_file = os.path.join(data_dir, f"{quarter}_stock_prices.jsonl")
    if not os.path.exists(stock_price_file):
        print(f"Stock price file not found for {quarter}")
        return

    # Load stock price data
    stock_price_dict = load_stock_price_data(stock_price_file)
    if not stock_price_dict:
        print(f"No stock price data for {quarter}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    output_file = os.path.join(output_base_dir, f"{quarter}_correlation.csv")

    # Store results for CSV
    results = []

    # Find all theme files
    theme_files = glob.glob(os.path.join(data_dir, f"*theme*.jsonl"))
    
    for theme_file in theme_files:
        if 'stock_prices' in theme_file:  # Skip stock price files
            continue
            
        # Extract theme name
        theme_name = os.path.basename(theme_file)
        theme_name = theme_name.replace(f"{quarter.lower()}_theme_", "")
        theme_name = theme_name.replace(".jsonl", "")

        print(f"\nProcessing theme: {theme_name}")
        
        # Process theme data
        merged_df = load_and_analyze_theme_data(theme_file, stock_price_dict, is_overall='overall' in theme_name)
        if merged_df is None:
            print(f"No theme data found in {theme_file}")
            continue

        # Filter top 50% by quote count
        df_filtered = filter_top_data_by_quotes(merged_df, percentage=0.5)
        if df_filtered is None:
            print(f"No filtered data for {theme_name}")
            continue

        # Calculate correlation
        correlation, p_value, _ = analyze_correlation(df_filtered)
        if correlation is None:
            continue

        # Store results
        results.append({
            'Theme': theme_name,
            'Correlation': correlation,
            'P_Value': p_value,
            'Sample_Size': len(df_filtered)
        })

    # Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print(f"No results to save for {quarter}")


def process_all_quarters(base_data_dir, output_base_dir):
    """
    Process all quarters in the base data directory.
    
    Args:
        base_data_dir: Base directory containing all quarter folders
        output_base_dir: Base directory for saving correlation results
    """
    # Find all quarter directories
    quarter_dirs = glob.glob(os.path.join(base_data_dir, "*_*Q"))
    
    for quarter_dir in sorted(quarter_dirs):
        print(f"\nProcessing quarter: {os.path.basename(quarter_dir)}")
        process_quarter_data(quarter_dir, output_base_dir)


def main():
    # Base directories
    base_data_dir = "/Users/junekwon/Desktop/Projects/scoring_agent/data/final/data"
    output_base_dir = "/Users/junekwon/Desktop/Projects/scoring_agent/data/final/figures/Corr"
    
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process all quarters
    process_all_quarters(base_data_dir, output_base_dir)


if __name__ == "__main__":
    main() 