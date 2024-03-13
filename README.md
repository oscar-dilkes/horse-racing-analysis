# Exploratory Analysis of Horse Racing Data
An exploratory analysis of horse racing data, achieved through the development of Python functions to achieve analyses of the provided dataset that are tailord but flexibile. The analysis uses common Python data science libraries, including pandas, matplotlib, and scipy.stats. The dataset undergoes a cleaning process, addressing missing values and converting incorrect formats. This code defines functions to calculate win data for specified columns, plot bar charts illustrating the relationship between selected columns and wins, and determine accuracy in predicting race outcomes based on forecasting and starting prices.
## Code Structure
### Data Cleaning
```python
def clean_data(df: pd.DataFrame):
```
- Cleans the input DataFrame by handling missing values and converting date formats.
- Drops columns with all missing values and rows with insufficient data for columns with fewer than 205 missing values.
- Provides feedback on dropped columns and rows.

### Utility Functions
```python
def convert_dates_to_odds(value):
```
- Converts day-month format to odds format.
```python
def string_odds_to_float(value):
```
- Cleans the input DataFrame by handling missing values and converting date formats.

### Win Data Calculation
```python
def win_data(df: pd.DataFrame, column_name: str, race_threshold: int, top_bottom_threshold: int):
```
- Calculates win data for specified columns, considering a threshold for the number of races and top/bottom entities.
- Provides information on the best and worst performers.

### Bar Chart Plotting
```python
def plot_bar_wins(df: pd.DataFrame, column_name: str):
```
- Plots a bar chart illustrating the relationship between specified columns and wins.
- Includes a regression line for better visualisation.
- Limited to 'WeightValue' and 'Age' columns due to regression line calculation.

### Accuracy Calculation
```python
def calculate_accuracy(df: pd.DataFrame, column_name: str):
```
- Calculates accuracy for specified odds columns.
- Compares odds of the winning horse with other horses and prints the accuracy percentage.

## Usage
```python
# Load raw data
raw_df = pd.read_csv("Horses.csv")

# Clean data
clean_df = clean_data(raw_df)

# Analyze and plot
win_data(clean_df, 'HorseID', 5, 5)
win_data(clean_df, 'JockeyID', 5, 5)
# ... (repeat for other columns)
plot_bar_wins(clean_df, 'Age')
plot_bar_wins(clean_df, 'WeightValue')
calculate_accuracy(clean_df, 'ForecastPrice')
calculate_accuracy(clean_df, 'StartingPrice')
```
