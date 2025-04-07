# Theme Analysis Tools

A collection of Python scripts for analyzing theme-based sentiment and stock market data.

## Data

### Data Download and Setup
1. Download the final dataset from [Google Drive](https://drive.google.com/file/d/1q92mhSl6O_7MTflZbkFt1D-3c-TU2CAw/view?usp=sharing)

2. Place the downloaded data in the following directory structure:

### Directory Structure
```
project/
└── data/
    └── final/
        └── data/
            ├── 2021_4Q/
            │   ├── 21_4q_theme_*.jsonl
            │   └── 2021_4Q_stock_prices.jsonl
            └── [other quarters...]
```

This directory structure ensures compatibility with all analysis scripts in the project. The data contains quarterly theme and stock price information in JSONL format.

### Output Directory
The project will generate and save analysis results in the following structure:
```
project/
└── figures/
    ├── CAR/
    │   └── [quarter_name]/
    │       └── [theme_name].png
    └── Corr/
        └── [quarter_name]_correlation.csv
```

Please ensure the `figures` directory exists before running the analysis scripts. The scripts will automatically create the necessary subdirectories (CAR and Corr) during execution.

## Scripts

### 1. car_time_series_analysis.py
Analyzes Cumulative Abnormal Returns (CAR) over time based on sentiment scores.

- Processes multiple quarters of theme data
- Calculates compound returns for accurate CAR measurement
- Generates time series plots comparing positive, negative, and overall sentiment groups
- Saves results as PNG files in the figures/CAR directory

### 2. correlation_analysis.py
Analyzes correlation between sentiment scores and short-term market reactions.

- Calculates correlation between sentiment scores and CAR(0,1)
- Processes multiple themes per quarter
- Generates CSV files with correlation statistics
- Saves results in the figures/Corr directory

### 3. top_filtered_themes.py
Identifies and analyzes top performing themes based on quote counts.

- Extracts top themes based on filtered quote count
- Provides detailed analysis of quote samples
- Generates summary statistics

## Input Data Format
- Theme files: JSONL format with filtered theme/overall output containing quotes and sentiment scores
- Stock price files: JSONL format with daily price and abnormal return data

## Output Format
### CAR Analysis
- PNG files showing 60-day CAR comparison between sentiment groups
- Includes sample sizes for each group (n=xx)

### Correlation Analysis
CSV files containing:
- Theme name (with sample size)
- Correlation coefficient
- P-value

## Usage

```python
# Process all quarters
python car_time_series_analysis.py

# Generate correlation analysis
python correlation_analysis.py

# Analyze top themes
python top_filtered_themes.py
```

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Notes
- For theme_overall files, the script uses 'filtered_overall_output' instead of 'filtered_theme_output'
- CAR calculations use compound returns: (1 + r1)(1 + r2)(1 + r3)... - 1
- Correlation analysis uses the top 25% of quotes by count
