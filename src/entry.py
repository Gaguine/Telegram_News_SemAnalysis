import argparse
import os, getpass
import pandas as pd
from Classes import Fetcher, Analyser  # Import your custom logic


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text Analysis CLI using LLMs.")

    # Positional command (run)
    parser.add_argument(
        'run',
        help='Starts the analysis')

    # Get the absolute path to the `Data` folder
    project_root = os.path.dirname(os.path.dirname(__file__))
    default_data_dir = os.path.join(project_root, "Data")

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
        '-re',
        '--restriction',
        type=int,
        nargs='?',
        default=5,
        help="'Maximum number of messages to process. Default is -1 (process all)."
    )

    """For Future Implementation"""
    # default_analysis_methods = ['classify_topic', 'sentiment_analysis']
    # parser.add_argument(
    #     '--analysis',
    #     nargs='+',
    #     choices=['classify_topic', 'sentiment_analysis', 'classify_sensitive_topic'],
    #     default=default_analysis_methods,
    #     help='Specify the analysis methods to use. Default: classify_topic and sentiment_analysis.')
    #
    parser.add_argument(
        '--output',
        default=os.path.join(os.path.dirname(__file__), "Output", "output.csv"),
        help='Path to the output CSV file. Defaults to "Output/output.csv".'
    )
    parser.add_argument(
        '--sep',
        default='|',
        help='CSV delimiter (default: "|").'
    )
    return parser.parse_args()


def run_analysis(html_files, output_file, sep, restriction):
    """Run the main analysis logic."""
    project_root = os.path.dirname(os.path.dirname(__file__))  # Get the root directory
    fetcher = Fetcher(project_root)
    analyser = Analyser()

    data = {
        "Date": [],
        "Semantic Tag": [],
        "Label": [],
        "Sensitive Topic": [],
    }

    for html_file in html_files:
        print(f"Processing file: {html_file}")
        fetcher.html_path = html_file  # Set the HTML path in Fetcher
        bs_messages = fetcher.read_html()
        message_list = fetcher.create_messages(bs_messages,restriction)

        for message in message_list:
            topic = analyser.classify_topic(message.text)
            sentiment = analyser.sentiment_analysis(message.text)
            sensitive_topic = analyser.classify_sensitive_topic(message.text)

            message.assign_new_labels(topic, sentiment, sensitive_topic)

            data["Date"].append(message.date)
            data["Semantic Tag"].append(message.sentiment)
            data["Label"].append(message.topic)
            data["Sensitive Topic"].append(message.sensitive_topic)

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    output_dir = os.path.join(project_root, "Output")
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    output_file_path = os.path.join(output_dir, "output.csv")  # Specify the file name
    df.to_csv(output_file_path, sep=sep, index=False)
    print(f"Analysis completed. Results saved to {output_file_path}")


def main():
    args = parse_args()

    if args.run == 'run':  # Check if the 'run' command is provided
        run_analysis(args.html, args.output, args.sep, args.restriction)
    else:
        print("Invalid command. Use 'run' to start the analysis.")


if __name__ == "__main__":
    main()
