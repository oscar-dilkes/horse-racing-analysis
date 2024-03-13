# Import necessary libraries
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def clean_data(df: pd.DataFrame):
    """
    Cleans the input DataFrame by handling missing values and converting date formats.
    """
    df['ForecastPrice'] = df['ForecastPrice'].apply(convert_dates_to_odds)
    df['StartingPrice'] = df['StartingPrice'].apply(convert_dates_to_odds)

    total_rows = df.shape[0]
    missing_data = df.isnull().sum()

    # Drop columns with all missing values
    columns_to_drop = missing_data[missing_data == total_rows].index
    df = df.drop(columns=columns_to_drop)
    print(f"Dropped the following columns as they contain no data: {columns_to_drop.tolist()}")

    indices_before_drop = df.index
    # Drop rows with missing data for columns with less than 205 missing values
    columns_to_consider = missing_data[missing_data < 205].index
    df = df.dropna(subset=columns_to_consider)
    indices_after_drop = df.index
    print(
        f"Dropped the following rows as they contained insufficient data: {indices_before_drop.difference(indices_after_drop).tolist()}")

    return df


def convert_dates_to_odds(value):
    """
    Converts day-month format to odds format.
    """
    try:
        date_obj = datetime.strptime(value, '%d-%b')
        return f"{date_obj.day}/{date_obj.month}"
    except Exception:
        return str(value)


def string_odds_to_float(value):
    """
    Converts string odds to float.
    """
    try:
        split_value = value.split('/')
        numerator, denominator = map(float, split_value)
        final_value = numerator / denominator
        return final_value
    except Exception:
        return None


def win_data(df: pd.DataFrame, column_name: str, race_threshold: int, top_bottom_threshold: int):
    """
    Calculates win data for specified columns.
    """
    valid_win_data_columns = ['HorseID', 'JockeyID', 'OwnerID', 'TrainerID',
                              'DamID', 'SireID', 'DamSireID', 'Sex', 'Colour']

    if column_name not in valid_win_data_columns:
        if column_name in df.columns.tolist():
            print(f"Unable to calculate win data '{column_name}' column.")
        else:
            print(f"Column '{column_name}' not present in dataframe.")
        return

    wins_df = df.groupby([column_name]).agg({'Won': sum})

    total_races_df = df.groupby([column_name]).agg({'RaceID': 'nunique'})
    total_races_df = total_races_df.rename(columns={'RaceID': 'TotalRaces'})

    wins_df = pd.merge(wins_df, total_races_df, left_index=True, right_index=True)

    # Filter out horses with fewer races than the specified threshold
    wins_df = wins_df[wins_df['TotalRaces'] >= race_threshold]

    proportional_wins_df = pd.DataFrame({
        'ProportionalWins': wins_df['Won'] / wins_df['TotalRaces'],
        'TotalRaces': wins_df['TotalRaces'].copy()
    })

    proportional_wins_df = proportional_wins_df.sort_values(by='TotalRaces', ascending=True)

    # Proportional Wins
    max_proportional_wins = proportional_wins_df['ProportionalWins'].max()
    best_proportional_df = proportional_wins_df.loc[proportional_wins_df['ProportionalWins'] == max_proportional_wins]
    top_x_proportional_df = proportional_wins_df.nlargest(top_bottom_threshold, 'ProportionalWins')

    proportional_wins_df = proportional_wins_df.sort_values(by='TotalRaces', ascending=False)

    min_proportional_wins = proportional_wins_df['ProportionalWins'].min()
    worst_proportional_df = proportional_wins_df.loc[proportional_wins_df['ProportionalWins'] == min_proportional_wins]
    bottom_x_proportional_df = proportional_wins_df.nsmallest(top_bottom_threshold, 'ProportionalWins')

    print(f"Best:\n{best_proportional_df}")
    print(f"Top {top_bottom_threshold}:\n{top_x_proportional_df}")
    print(f"Worst:\n{worst_proportional_df}")
    print(f"Bottom {top_bottom_threshold}:\n{bottom_x_proportional_df}")


def plot_bar_wins(df: pd.DataFrame, column_name: str):
    """
    Plots bar chart showing the relationship between specified column and wins.
    """
    if column_name != 'WeightValue' and column_name != 'Age':
        if column_name not in df.columns.tolist():
            print(f"Unable to plot chart for '{column_name}' column.")
        else:
            print(f"Column '{column_name}' not present in dataframe.")
        return

    horse_weight_wins_df = df.groupby([column_name]).agg({'Won': ['sum', 'count']}).reset_index()
    horse_weight_wins_df.columns = [column_name, 'TotalWins', 'TotalHorses']
    horse_weight_wins_df['WeightedAverage'] = horse_weight_wins_df['TotalWins'] / horse_weight_wins_df['TotalHorses']

    plt.bar(horse_weight_wins_df[column_name], horse_weight_wins_df['WeightedAverage'])
    plt.xlabel(column_name)
    plt.ylabel('Weighted Average of Wins')
    plt.title('Relationship Between ' + column_name + ' and Wins')

    slope, intercept, _, p_value, _ = linregress(horse_weight_wins_df[column_name],
                                                 horse_weight_wins_df['WeightedAverage'])

    regression_line = slope * horse_weight_wins_df[column_name] + intercept

    plt.plot(horse_weight_wins_df[column_name], regression_line, color='red', label='Regression Line')

    plt.show()

    if p_value < 0.05:
        print(f"The slope is statistically significant at α = 0.05 (p-value: {p_value:.4f}).")
    else:
        print(f"The slope is not statistically significant at α = 0.05 (p-value: {p_value:.4f}).")


def calculate_accuracy(df: pd.DataFrame, column_name: str):
    """
    Calculates accuracy for specified odds columns.
    """
    if column_name != 'ForecastPrice' and column_name != 'StartingPrice':
        if column_name in df.columns.tolist():
            print(f"Unable to calculate price accuracy for '{column_name}' column.")
        else:
            print(f"Column '{column_name}' not present in dataframe.")
        return

    df[column_name] = df[column_name].apply(string_odds_to_float)

    # If Won == 1, then that's the winner of the race
    actual_winners = df[df['Won'] == 1][['RaceID', 'HorseID', column_name]]
    race_odds = pd.merge(actual_winners, df[['RaceID', 'HorseID', column_name]], on='RaceID')

    # Count the number of correct predictions where the odds for the horse in column_name + '_x'
    # are less than or equal to the odds for the same horse in column_name + '_y'
    print(race_odds[column_name + '_x'])
    print(race_odds[column_name + '_y'])
    correct_predictions = (race_odds[column_name + '_x'] <= race_odds[column_name + '_y']).sum()
    total_predictions = race_odds.shape[0]

    accuracy = correct_predictions / total_predictions

    # Print the accuracy of predictions for the specified column_name, formatted as a percentage to two dp
    print(f"{column_name} Accuracy: {accuracy:.2%}")


# Load raw data
raw_df = pd.read_csv("Horses.csv")

# Clean data
clean_df = clean_data(raw_df)

calculate_accuracy(clean_df, "ForecastPrice")
