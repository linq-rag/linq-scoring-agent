import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter


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
                    # Sort by date (from oldest)
                    stock_prices.sort(key=lambda x: x.get('date', ''))

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


def load_theme_data(theme_file):
    """
    Load theme data file and extract necessary information.
    """
    theme_data = []

    try:
        with open(theme_file, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())

                    # Extract necessary data
                    custom_id = data.get('custom_id', '')
                    
                    # Check if this is an overall theme file
                    if 'overall' in theme_file:
                        filtered_quotes = data.get('filtered_overall_output', {}).get('quotes', [])
                        filtered_scores = data.get('filtered_overall_output', {}).get('sentiment_scores', [])
                    else:
                        filtered_quotes = data.get('filtered_theme_output', {}).get('quotes', [])
                        filtered_scores = data.get('filtered_theme_output', {}).get('sentiment_scores', [])

                    # Skip if no quotes
                    if not filtered_quotes:
                        continue

                    # Extract ticker and event date
                    ticker = 'UNKNOWN'
                    event_date = None

                    if custom_id.startswith('task-'):
                        parts = custom_id.split('-')
                        if len(parts) > 1:
                            ticker = parts[1]

                        if len(parts) >= 4:  # task-TICKER-YY-MM-DD format
                            year = '20' + parts[2]  # Convert YY to YYYY
                            month = parts[3]
                            day = parts[4].split('_')[0]  # Extract DD from DD_ format
                            event_date = f"{year}-{month}-{day}"

                    # Skip if scores are empty or length is different from quotes
                    if not filtered_scores or len(filtered_scores) != len(filtered_quotes):
                        continue

                    # Calculate average sentiment score
                    avg_sentiment_score = np.mean(filtered_scores)

                    theme_data.append(
                        {
                            'ticker': ticker,
                            'custom_id': custom_id,
                            'event_date': event_date,
                            'filtered_count': len(filtered_quotes),
                            'avg_sentiment_score': avg_sentiment_score
                        }
                    )

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing data: {str(e)}")
                    continue

        print(f"Theme data loaded: {len(theme_data)} data loaded")
        return theme_data

    except Exception as e:
        print(f"Error processing theme data file: {str(e)}")
        return []


def filter_by_quote_count(theme_data, percentage=0.5):
    """
    Extract the top percentage% of data based on the number of filtered quotes (1 or more).
    """
    # Filter data with 1 or more filtered quotes
    valid_data = [item for item in theme_data if item['filtered_count'] >= 1]

    # Sort by number of filtered quotes in descending order
    sorted_data = sorted(valid_data, key=lambda x: x['filtered_count'], reverse=True)

    # Extract top percentage%
    top_count = max(1, int(len(sorted_data) * percentage))
    filtered_data = sorted_data[:top_count]

    print(f"Filtered data based on quote count, top {percentage * 100}%: {len(filtered_data)} / {len(theme_data)}")

    return filtered_data


def find_event_index(stock_prices, event_date):
    """
    Find the index of the event date in the stock price data.
    """
    for i, price_data in enumerate(stock_prices):
        if price_data.get('date') == event_date:
            return i
    return None


def calculate_car_series(stock_prices, event_index, window=60):
    """
    Calculate the cumulative abnormal return (CAR) series from the event date for a specified period.
    Using compound returns instead of simple sum.
    """
    if event_index is None or event_index >= len(stock_prices):
        return None

    car_series = []
    cumulative_ar = 1.0  # Start with 1 (100%)

    # Calculate CAR from event date to window days
    days_counted = 0
    i = event_index

    while days_counted <= window and i < len(stock_prices):
        price_data = stock_prices[i]
        ar = price_data.get('abnormal_return')

        if ar is not None:
            # Compound return: (1 + r1)(1 + r2)(1 + r3)... - 1
            cumulative_ar *= (1 + ar)
            car_series.append(cumulative_ar - 1)  # Convert to return percentage
            days_counted += 1

        i += 1

    # Pad to match window length
    if len(car_series) <= window:
        last_value = car_series[-1] if car_series else 0
        car_series.extend([last_value] * (window + 1 - len(car_series)))

    return car_series[:window + 1]  # CAR from 0 to window days


def process_data_for_car_analysis(theme_data, stock_price_dict, window=60):
    """
    Combine theme data and stock price data to prepare for CAR analysis.
    """
    car_data = []

    for data in theme_data:
        ticker = data['ticker']
        event_date = data['event_date']

        # Check if ticker and event date are valid
        if ticker == 'UNKNOWN' or not event_date or ticker not in stock_price_dict:
            continue

        stock_prices = stock_price_dict[ticker]['stock_prices']

        # Find event date index
        event_index = find_event_index(stock_prices, event_date)
        if event_index is None:
            continue

        # Calculate CAR series
        car_series = calculate_car_series(stock_prices, event_index, window)
        if car_series is None or len(car_series) <= window:
            continue

        # Save result
        car_data.append(
            {
                'ticker': ticker,
                'event_date': event_date,
                'filtered_count': data['filtered_count'],
                'avg_sentiment_score': data['avg_sentiment_score'],
                'car_series': car_series
            }
        )

    print(f"CAR series calculation completed: {len(car_data)} data processed")
    return car_data


def split_data_by_sentiment_percentile(car_data):
    """
    Classify data into top 25% and bottom 25% groups based on sentiment score.
    """
    # Sort by sentiment score
    sorted_data = sorted(car_data, key=lambda x: x['avg_sentiment_score'])

    # Total data count
    total_count = len(sorted_data)

    # Calculate bottom 25% and top 25% cutoff indices
    bottom_25_cutoff = max(1, int(total_count * 0.25))
    top_25_cutoff = max(1, int(total_count * 0.75))

    # Group classification
    bottom_25_pct = sorted_data[:bottom_25_cutoff]
    top_25_pct = sorted_data[top_25_cutoff:]

    print(f"Top 25% sentiment score group size: {len(top_25_pct)}")
    print(f"Bottom 25% sentiment score group size: {len(bottom_25_pct)}")

    return top_25_pct, bottom_25_pct


def calculate_average_car_by_group(top_group, bottom_group, all_data, window=60):
    """
    Calculate the average CAR series for each group and all data.
    """
    # Calculate average CAR for all data
    all_cars = np.zeros(window + 1)
    for data in all_data:
        all_cars += np.array(data['car_series'])
    
    if len(all_data) > 0:
        all_avg_car = all_cars / len(all_data)
    else:
        all_avg_car = np.zeros(window + 1)

    # Calculate average CAR for top group
    top_cars = np.zeros(window + 1)
    for data in top_group:
        top_cars += np.array(data['car_series'])

    if len(top_group) > 0:
        top_avg_car = top_cars / len(top_group)
    else:
        top_avg_car = np.zeros(window + 1)

    # Calculate average CAR for bottom group
    bottom_cars = np.zeros(window + 1)
    for data in bottom_group:
        bottom_cars += np.array(data['car_series'])

    if len(bottom_group) > 0:
        bottom_avg_car = bottom_cars / len(bottom_group)
    else:
        bottom_avg_car = np.zeros(window + 1)

    return top_avg_car, bottom_avg_car, all_avg_car


def plot_avg_car_comparison(top_avg_car, bottom_avg_car, all_avg_car, window=60, output_path=None, top_n=0, bottom_n=0, all_n=0):
    """
    Plot a line graph comparing the average CAR of all groups.
    """
    plt.figure(figsize=(12, 8))

    # Create series and plot graph
    days = np.arange(window + 1)

    # Set graph style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'  # Set English font

    # Line plot with sample size in labels
    plt.plot(days, all_avg_car, 'k-', linewidth=2, label=f'All (n={all_n})')
    plt.plot(days, top_avg_car, 'g-', linewidth=2, label=f'Positive Group (n={top_n})')
    plt.plot(days, bottom_avg_car, 'r-', linewidth=2, label=f'Negative Group (n={bottom_n})')

    # Add 0 line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Set graph
    plt.title('Sentiment Score-based 60-Day Cumulative Abnormal Return (CAR) Comparison', fontsize=16)
    plt.xlabel('Days since Event', fontsize=14)
    plt.ylabel('Average Cumulative Abnormal Return (CAR)', fontsize=14)

    # Display y-axis as percentage format
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Adjust legend position
    plt.legend(loc='best', fontsize=12)

    plt.tight_layout()

    # Save graph
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    
    plt.close()  # Close the figure to free memory


def process_quarter_data(data_dir, output_base_dir):
    """
    Process all theme files in a quarter directory and generate CAR analysis graphs.
    
    Args:
        data_dir: Directory containing theme and stock price files (e.g., '.../2021_4Q')
        output_base_dir: Base directory for saving figures (e.g., '.../figures/CAR')
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
    output_dir = os.path.join(output_base_dir, quarter)
    os.makedirs(output_dir, exist_ok=True)

    # Find all theme files
    theme_files = glob.glob(os.path.join(data_dir, f"*theme*.jsonl"))
    
    for theme_file in theme_files:
        if 'stock_prices' in theme_file:  # Skip stock price files
            continue
            
        # Extract theme name for output file
        theme_name = os.path.basename(theme_file)
        theme_name = theme_name.replace(f"{quarter.lower()}_theme_", "")
        theme_name = theme_name.replace(".jsonl", "")
        output_file = os.path.join(output_dir, f"{theme_name}.png")

        print(f"\nProcessing theme: {theme_name}")
        
        # Process theme data
        theme_data = load_theme_data(theme_file)
        if not theme_data:
            print(f"No theme data found in {theme_file}")
            continue

        # Filter and process data
        filtered_theme_data = filter_by_quote_count(theme_data, percentage=0.5)
        if not filtered_theme_data:
            print(f"No filtered data for {theme_name}")
            continue

        car_data = process_data_for_car_analysis(filtered_theme_data, stock_price_dict, window=60)
        if not car_data:
            print(f"No CAR data for {theme_name}")
            continue

        # Generate and save graph
        top_group, bottom_group = split_data_by_sentiment_percentile(car_data)
        top_avg_car, bottom_avg_car, all_avg_car = calculate_average_car_by_group(
            top_group, 
            bottom_group,
            car_data,  # 전체 데이터 전달
            window=60
        )
        plot_avg_car_comparison(
            top_avg_car, 
            bottom_avg_car,
            all_avg_car,
            output_path=output_file,
            top_n=len(top_group),
            bottom_n=len(bottom_group),
            all_n=len(car_data)  # 전체 데이터 개수 전달
        )

        print(f"Generated graph for {theme_name}")
        print(f"Saved to: {output_file}")


def process_all_quarters(base_data_dir, output_base_dir):
    """
    Process all quarters in the base data directory.
    
    Args:
        base_data_dir: Base directory containing all quarter folders
        output_base_dir: Base directory for saving figures
    """
    # Find all quarter directories
    quarter_dirs = glob.glob(os.path.join(base_data_dir, "*_*Q"))
    
    for quarter_dir in sorted(quarter_dirs):
        print(f"\nProcessing quarter: {os.path.basename(quarter_dir)}")
        process_quarter_data(quarter_dir, output_base_dir)


def main():
    # Base directories
    base_data_dir = "/Users/junekwon/Desktop/Projects/scoring_agent/data/final/data"
    output_base_dir = "/Users/junekwon/Desktop/Projects/scoring_agent/data/final/figures/CAR"
    
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process all quarters
    process_all_quarters(base_data_dir, output_base_dir)


if __name__ == "__main__":
    main()
