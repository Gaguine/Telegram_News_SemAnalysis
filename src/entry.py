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

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text Analysis CLI using LLMs.")

    # Paths for Data and Output folders
    project_root = os.getcwd()
    default_data_dir = os.path.join(project_root, "Data")
    default_output_dir = os.path.join(project_root, "Output")

    if not os.path.exists(default_data_dir):
        raise FileNotFoundError(f"Default data directory not found: {default_data_dir}")

    # Automatically list all `.html` files in the `Data` directory
    default_html_files = [
        os.path.join(default_data_dir, f) for f in os.listdir(default_data_dir) if f.endswith(".html")
    ]

    parser.add_argument('run', help='Starts the analysis')
    parser.add_argument('visualize', help='Visualize the csv output.')

    # Graph options
    parser.add_argument("-gt", "--general_timeline", help="Create a general timeline from the csv output.")
    parser.add_argument("-tt", "--topic_timeline", help="Create topic timeline from the csv output.", default=None)
    parser.add_argument("-gh", "--general_histogram", help="Create a general histogram from the csv output.")
    parser.add_argument("-th", "--topic_histogram", help="Create a topic histogram from the csv output.")
    parser.add_argument("-tdt", "--topic_dynamics_timeline", help="Create a dynamics timeline for all topics.")
    parser.add_argument("-tfh", "--topic_frequency_hist", help="Create a topic frequency histogram.")

    parser.add_argument('--html', nargs='+', default=default_html_files,
                        help='Path(s) to input HTML file(s). Defaults to files in the "Data" folder.')
    parser.add_argument('--output', default=os.path.join(default_output_dir, "output.csv"),
                        help='Path to the output CSV file. Defaults to "Output/output.csv".')
    parser.add_argument('--sep', default='|', help='CSV delimiter (default: "|").')
    parser.add_argument('-re', '--restriction', type=int, default=-1,
                        help="Maximum number of messages to process. Default is -1.")
    parser.add_argument('-ft', '--filter-topic',
                        help="Filter results by topic. Possible topics: Politics, Economy, Technology, etc.")

    return parser.parse_args()

def run_analysis(html_files, restriction) -> pd.DataFrame:
    """Run the main analysis logic."""
    project_root = os.getcwd()
    fetcher = Fetcher(project_root)
    analyser = Analyser()

    data = {
        "Date": [],
        "Semantic Tag": [],
        "Label": [],
        # "Sensitive Topic": [],
    }

    for html_file in html_files:
        print(f"Processing file: {html_file}")
        fetcher.html_path = html_file
        bs_messages = fetcher.read_html()
        message_list = fetcher.create_messages(bs_messages, restriction)

    print(f"Collected {len(message_list)} messages. Commencing LLM analysis")

    # do a step-by-step analysis: Begin with semantic tag
    analyser.load_sentiment_model()
    for message in message_list:

        sentiment = analyser.sentiment_analysis(message.text)
        # sensitive_topic = analyser.classify_sensitive_topic(message.text)
        message.assign_sentiment(sentiment)
        data["Date"].append(message.date)
        data["Semantic Tag"].append(message.sentiment)
    analyser.clear_models()
    print("Semantic analysis completed")

    # start topic analysis
    analyser.load_topic_model()
    for message in message_list:
        topic = analyser.classify_topic(message.text)
        message.assign_topic(topic)
        data["Label"].append(message.topic)
        # data["Sensitive Topic"].append(message.sensitive_topic)
    analyser.clear_models()
    print("Topic analysis completed")
    # Add Sensitive Topic Analysis
    print("Analysis completed.")
    return pd.DataFrame(data)
def display_output(data:pd.DataFrame,output_file, sep):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Create a DataFrame and save it to a CSV file
    displayer = Displayer(data)
    displayer.create_csv(output_file,sep)
def display_graph(data: pd.DataFrame, args, topic_list=None):
    """Handles graph creation and visualization logic."""
    displayer = Displayer(data)

    if args.general_timeline:
        plt_obj = displayer.create_general_timeline(data)
        if plt_obj:
            output_path = os.path.join(args.general_timeline)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt_obj.savefig(output_path)
            print(f"General timeline saved at {output_path}")

    if args.general_histogram:
        plt_obj = displayer.create_general_hist(data)
        if plt_obj:
            output_path = os.path.join(args.general_histogram)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt_obj.savefig(output_path)
            print(f"General histogram saved at {output_path}")

    if args.topic_timeline:
        plt_obj = displayer.create_timeline_by_topic(args.topic_timeline, data)
        if plt_obj:
            output_path = os.path.join(args.topic_timeline)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt_obj.savefig(output_path)
            print(f"Topic timeline saved at {output_path}")

    if args.topic_dynamics_timeline:
        if topic_list is None:
            topic_list = list(data['Label'].unique())
        plt_obj = displayer.create_topic_dynamics_timeline(topic_list, data)
        if plt_obj:
            output_path = os.path.join(args.topic_dynamics_timeline)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt_obj.savefig(output_path)
            print(f"Topic dynamics timeline saved at {output_path}")

    if args.topic_frequency_hist:
        if topic_list is None:
            topic_list = list(data['Label'].unique())
        plt_obj = displayer.create_topic_frequency_hist(topic_list, data)
        if plt_obj:
            output_path = os.path.join(args.topic_frequency_hist)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt_obj.savefig(output_path)
            print(f"Topic frequency histogram saved at {output_path}")


def main():
    show_sick_banner()
    args = parse_args()

    if args.run == 'run':  # Check if the 'run' command is provided
        data = run_analysis(args.html, args.restriction)

        filter = Filter()

        if args.filter_topic:
            print(f"Filtering by topic: {args.filter_topic}")
            data = filter.filter_data(data, user_topic=args.filter_topic)

        display_output(data, args.output, args.sep)

    elif args.run == 'visualize':  # Check if the 'visualize' command is provided
        if os.path.exists(args.output):
            print(f"Loading data from {args.output}...")
            data = pd.read_csv(args.output, sep=args.sep)
            topic_list = list(set(data["Label"]))
            display_graph(data, args,topic_list=topic_list)
        else:
            print(f"CSV output file not found: {args.output}. Please run analysis first.")
    else:
        print("Invalid command. Use 'run' to start the analysis. Use 'visualize' to create timeline or histogram.")


if __name__ == "__main__":
    main()
