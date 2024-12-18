import argparse
import os
import pandas as pd
from Classes import Fetcher, Analyser, Displayer, Filter


def show_sick_banner():
    banner = """
························································
:                                                      :
:  ░█▀█░█▀▀░█░█░█▀▀░░░█▀█░█▀█░█▀█░█░░░█░█░█▀▀░▀█▀░█▀▀  :
:  ░█░█░█▀▀░█▄█░▀▀█░░░█▀█░█░█░█▀█░█░░░░█░░▀▀█░░█░░▀▀█  :
:  ░▀░▀░▀▀▀░▀░▀░▀▀▀░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀▀▀  :
:                                                      :
························································"""
    print(banner)

# Create commands for the CLI
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text Analysis CLI using LLMs.")

    # Subcommands for 'run' and 'visualize'
    subparsers = parser.add_subparsers(dest='command', help="Command to execute (run or visualize)")

    # 'run' subcommand
    run_parser = subparsers.add_parser('run', help='Run the analysis on all HTML files in the Data folder.')
    run_parser.add_argument("-re", "--restriction", type=int, default=-1,
                            help="Number of messages to analyze. Use -1 for all messages.")

    # 'visualize' subcommand
    visualize_parser = subparsers.add_parser('visualize', help='Visualize the analysis output.')
    visualize_parser.add_argument('-gt', '--general_timeline', action='store_true',
                                  help='Create a general timeline of the Semantic Tag from the csv output.')
    visualize_parser.add_argument('-tdt', '--topic_dynamics_timeline', action='store_true',
                                  help='Create topic popularity dynamics timeline from the csv output.') #Topic dynamics
    visualize_parser.add_argument('-gh', '--general_histogram', action='store_true',
                                  help='Create a general histogram of the Semantic Tag from the csv output.')
    visualize_parser.add_argument('-tt', '--topic_timeline', type=str,
                                  help='Create a semantic tag timeline for a specific topic.')
    visualize_parser.add_argument('-th', '--topic_histogram', type=str,
                                  help='Create a topic histogram for a specific topic.')
    visualize_parser.add_argument('-tfh', '--topic_frequency_hist', action='store_true',
                                  help='Create a topic frequency histogram from the csv output.')

    return parser.parse_args()

# Assign actions and logic to the CLI commands
def run_analysis(restriction: int) -> pd.DataFrame:
    """
    Executes the main analysis logic, including fetching messages,
    performing sentiment analysis, and classifying topics.

    This function integrates various components (Fetcher, Analyser) to process
    messages from HTML files, analyze their sentiment, and classify them into topics.

    Args:
        restriction (int): The maximum number of messages to process. Use -1 for no restriction.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data with columns:
                      - Date
                      - Semantic Tag
                      - Label(Topic)
                      - Sensitive Topic(currently not implemented)
    """
    fetcher = Fetcher(os.getcwd())
    analyser = Analyser()

    # Fetch and process messages directly using Fetcher
    bs_messages = fetcher.read_html()
    if not bs_messages:
        print("No messages found in the HTML files. Exiting analysis.")
        return pd.DataFrame()

    message_list = fetcher.create_messages(bs_messages, restriction)

    if not message_list:
        print("No valid messages found. Exiting analysis.")
        return pd.DataFrame()

    print(f"Collected {len(message_list)} messages. Commencing LLM analysis.")

    # Prepare data storage
    data = {
        "Date": [],
        "Semantic Tag": [],
        "Label": [],
    }

    # Perform sentiment analysis
    analyser.load_sentiment_model()
    for message in message_list:
        sentiment = analyser.sentiment_analysis(message.text)
        message.assign_sentiment(sentiment)
        data["Date"].append(message.date)
        data["Semantic Tag"].append(message.sentiment)
    analyser.clear_models()
    print("Semantic analysis completed.")

    # Perform topic analysis
    analyser.load_topic_model()
    for message in message_list:
        topic = analyser.classify_topic(message.text)
        message.assign_topic(topic)
        data["Label"].append(message.topic)
    analyser.clear_models()
    print("Topic analysis completed.")

    print("Analysis completed.")
    return pd.DataFrame(data)
def display_output(data:pd.DataFrame,output_file, sep):
    """
    Create the output.csv file

    Args:
    data (pd.DataFrame): The processed data to be saved.
    output_file (str): Path to the output CSV file.
    sep (str): Delimiter to use in the CSV file.
    """
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Create a DataFrame and save it to a CSV file
    displayer = Displayer(data)
    displayer.create_csv(output_file,sep)
def display_graph(data: pd.DataFrame, args, topic_list=None):
    """
    Handles graph creation and visualization logic.
     Args:
        data (pd.DataFrame): The processed data used for visualization.
        args: Parsed arguments specifying which graphs to create.
        topic_list (list, optional): List of topics for topic-specific visualizations.
                                     Defaults to None.

    The function creates and saves:
        - General timeline
        - General histogram
        - Topic dynamics timeline
        - Topic frequency histogram
        - Topic-specific timelines and histograms
    """
    displayer = Displayer(data)
    output_dir = os.path.join(os.getcwd(), "Output")  # Ensure files are saved in the "Output" folder
    os.makedirs(output_dir, exist_ok=True)

    if args.general_timeline:
        print("Creating general timeline...")
        plt_obj = displayer.create_general_timeline(data)
        output_path = os.path.join(output_dir, "general_timeline.png")
        if plt_obj:
            plt_obj.savefig(output_path)
            print(f"General timeline saved at {output_path}")
        else:
            print("Failed to create general timeline.")
    if args.general_histogram:
        print("Creating general histogram...")
        plt_obj = displayer.create_general_hist(data)
        output_path = os.path.join(output_dir, "general_histogram.png")
        if plt_obj:
            plt_obj.savefig(output_path)
            print(f"General histogram saved at {output_path}")
        else:
            print("Failed to create general histogram.")
    if args.topic_dynamics_timeline:
        print("Creating topic timeline...")
        if topic_list is None:
            topic_list = list(data['Label'].unique())
        plt_obj = displayer.create_topic_dynamics_timeline(topic_list, data)
        output_path = os.path.join(output_dir, "topic_timeline.png")
        if plt_obj:
            plt_obj.savefig(output_path)
            print(f"Topic timeline saved at {output_path}")
        else:
            print("Failed to create topic timeline.")
    if args.topic_frequency_hist:
        print("Creating topic frequency histogram...")
        if topic_list is None:
            topic_list = list(data['Label'].unique())
        plt_obj = displayer.create_topic_frequency_hist(topic_list, data)
        output_path = os.path.join(output_dir, "topic_frequency_histogram.png")
        if plt_obj:
            plt_obj.savefig(output_path)
            print(f"Topic frequency histogram saved at {output_path}")
        else:
            print("Failed to create topic frequency histogram.")
    if args.topic_timeline:
        print(f"Creating topic timeline for topic: {args.topic_timeline}")
        plt_obj = displayer.create_timeline_by_topic(args.topic_timeline, data)
        output_path = os.path.join(output_dir, f"{args.topic_timeline}_timeline.png")
        if plt_obj:
            plt_obj.savefig(output_path)
            print(f"Topic timeline for {args.topic_timeline} saved at {output_path}")
        else:
            print(f"Failed to create topic timeline for {args.topic_timeline}.")
    if args.topic_histogram:
        print(f"Creating topic histogram for topic: {args.topic_histogram}")
        plt_obj = displayer.create_hist_by_topic(args.topic_histogram, data)
        output_path = os.path.join(output_dir, f"{args.topic_histogram}_histogram.png")
        if plt_obj:
            plt_obj.savefig(output_path)
            print(f"Topic histogram for {args.topic_histogram} saved at {output_path}")
        else:
            print(f"Failed to create topic histogram for {args.topic_histogram}.")


def main():
    show_sick_banner()
    args = parse_args()

    # Default output file
    output_dir = os.path.join(os.getcwd(), "Output")
    output_file = os.path.join(output_dir, "output.csv")
    os.makedirs(output_dir, exist_ok=True)

    if args.command == 'run':  # Automatically process all HTML files in the Data folder
        print("Starting analysis...")
        data = run_analysis(restriction=args.restriction)

        if data.empty:
            print("No data to process. Exiting.")
            return

        display_output(data, output_file, sep="|")
        print(f"Analysis complete. Results saved in {output_file}")

    elif args.command == 'visualize':  # Visualization logic
        if os.path.exists(output_file):
            print(f"Loading data from {output_file}...")
            data = pd.read_csv(output_file, sep="|")
            topic_list = list(data['Label'].unique())
            display_graph(data, args, topic_list=topic_list)
        else:
            print(f"CSV output file not found: {output_file}. Please run analysis first.")
    else:
        print("Invalid command. Use 'run' to start the analysis or 'visualize' to create graphs.")


if __name__ == "__main__":
    main()