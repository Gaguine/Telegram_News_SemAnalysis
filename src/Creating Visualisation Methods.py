import pandas as pd
from matplotlib import pyplot as plt


def create_timeline_by_topic(user_topic, data: pd.DataFrame):
    try:
        # Check if the user_topic exists in the dataset
        if user_topic in set(data["Label"]):
            filtered_data = data[data['Label'] == user_topic]

            # Convert Date column to datetime
            filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], format='%d/%m/%Y')

            # Map semantic tags to numeric values
            semantic_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
            filtered_data.loc[:, 'Semantic Tag Encoded'] = filtered_data['Semantic Tag'].map(semantic_map)

            # Group by date and calculate the sum
            grouped_data = filtered_data.groupby('Date')['Semantic Tag Encoded'].sum().reset_index()

            # Plot the timeline
            plt.figure(figsize=(12, 6))
            plt.plot(grouped_data['Date'], grouped_data['Semantic Tag Encoded'], marker='o', linestyle='-')
            plt.title(f"Semantic Tags Timeline for Topic: {user_topic}")
            plt.xlabel("Date")
            plt.ylabel("Semantic Tags")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Topic '{user_topic}' not found in the data.")
    except KeyError as k:
        print(f"KeyError: {k}. Ensure the column names are correct.")
def create_hist_by_topic(user_topic, data: pd.DataFrame):
    try:
        # Check if the user_topic exists in the dataset
        if user_topic in set(data["Label"]):
            filtered_data = data[data['Label'] == user_topic]

            # Map semantic tags to numeric values
            semantic_counts = filtered_data['Semantic Tag'].value_counts()

            # Plot the histogram
            plt.figure(figsize=(8, 5))
            plt.bar(semantic_counts.index, semantic_counts.values, color='skyblue', edgecolor='black')
            plt.title(f"Frequency of Semantic Tags for Topic: {user_topic}")
            plt.xlabel("Semantic Tag")
            plt.ylabel("Frequency")
            plt.xticks(['Negative', 'Neutral', 'Positive'])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Topic '{user_topic}' not found in the data.")
    except KeyError as k:
        print(f"KeyError: {k}. Ensure the column names are correct.")
def create_timeline(data: pd.DataFrame):
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    # Map semantic tags to numeric values
    semantic_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    data.loc[:, 'Semantic Tag Encoded'] = data['Semantic Tag'].map(semantic_map)

    # Group by month and calculate the sum
    data['Month'] = data['Date'].dt.to_period('M')  # Extract year-month period
    grouped_data = data.groupby('Month')['Semantic Tag Encoded'].sum().reset_index()

    # Convert the Month period back to a datetime object for plotting
    grouped_data['Month'] = grouped_data['Month'].dt.to_timestamp()

    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    # Map semantic tags to numeric values
    semantic_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    data.loc[:, 'Semantic Tag Encoded'] = data['Semantic Tag'].map(semantic_map)

    # Group by date and calculate the sum
    grouped_data = data.groupby('Date')['Semantic Tag Encoded'].sum().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(grouped_data['Date'], grouped_data['Semantic Tag Encoded'], marker='o', linestyle='-')

    # Format the x-axis to show only the beginning of each month
    start_of_months = grouped_data['Date'].dt.to_period('M').drop_duplicates().dt.to_timestamp()
    plt.xticks(start_of_months, labels=start_of_months.dt.strftime('%b %Y'), rotation=45)

    # Add titles and labels
    plt.title("Semantic Tags Timeline (By Day, Monthly Labels)")
    plt.xlabel("Date")
    plt.ylabel("Sum of Semantic Tags")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

user_topic = "politics"
data = pd.read_csv("output (3).csv",sep='|')



