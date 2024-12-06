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

    # Positional command (run)
    parser.add_argument(
        'run', help='Starts the analysis')

    # Paths for Data and Output folders
    project_root = os.getcwd()
    default_data_dir = os.path.join(project_root, "Data")
    default_output_dir = os.path.join(project_root, "Output")

    # Check if the `Data` directory exists
    if not os.path.exists(default_data_dir):
        raise FileNotFoundError(f"Default data directory not found: {default_data_dir}")

    # Automatically list all `.html` files in the `Data` directory
    default_html_files = [
        os.path.join(default_data_dir, f) for f in os.listdir(default_data_dir) if f.endswith(".html")
    ]

    parser.add_argument(
        '--html',
        nargs='+',
        default=default_html_files,
        help='Path(s) to input HTML file(s). Defaults to files in the "Data" folder.'
    )
    parser.add_argument(
        '--output',
        default=os.path.join(default_output_dir, "output.csv"),
        help='Path to the output CSV file. Defaults to "Output/output.csv".'
    )
    parser.add_argument(
        '--sep',
        default='|',
        help='CSV delimiter (default: "|").'
    )
    parser.add_argument(
        '-re',
        '--restriction',
        type=int,
        default=-1,
        help="Maximum number of messages to process. Default is -1."
    )
    parser.add_argument('-ft','--filter-topic',
                        help="Filter results by topic. Possible topics:"
                             "Politics, Economy, Technology, Sports, Health, Entertainment, Science, Environment,World News, Local News")
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
    else:
        print("Invalid command. Use 'run' to start the analysis.")


if __name__ == "__main__":
    main()
