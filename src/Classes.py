import os, torch, json
from bs4 import BeautifulSoup
from transformers import pipeline,AutoTokenizer, AutoModelForSequenceClassification
from matplotlib import pyplot as plt
import pandas as pd


class TgMessage():
    """This class will create objects resembling telegram messages, but with a dictionary-like structure. TgMessage objects will store the "meta" data of a particular message/news. Specifically the:
    -TEXT CONTENT
    -DATE
    -TOPIC
    -SEMANTIC TAG(or sentiment)

    Every TgMessage object has the following attributes:

    WHEN FETCHING THE DATA FROM THE HTML
    self.text = text
    self.date = date

    AFTER COMPLETING THE LLM ANALYSIS
    self.topic = topic
    self.sentiment = sentiment
    self.sensitive_topic = sensitive_topic

    STORE THE ATTRIBUTES IN A PYTHON DICTIONARY FOR PD.DF CONVERSION
    self.contents = {
                    "text":self.text,
                    "date":self.date,
                    "topic":self.topic,
                    "sentiment":self.sentiment (this is the semantic tag)
                    "sensitive_topic":self.sensitive_topic
    }

    """

    # initialize all attribute to avoid undesired IDE errors
    def __init__(self,text:str,date):
        self.text = text
        self.date = date
        self.contents={}
        self.topic = None
        self.sentiment = None
        self.sensitive_topic = None

    # Update obj attributes after analysis
    def assign_topic(self,topic:str):
        self.topic = topic
    def assign_sentiment(self,sentiment:str):
        self.sentiment = sentiment
    def assign_sensitive_topic(self,sensitive_topic:str):
        self.sensitive_topic = sensitive_topic
    def assign_new_labels(self,topic:str,sentiment:str,sensitive_topic:str):
        self.assign_topic(topic)
        self.assign_sentiment(sentiment)
        self.assign_sensitive_topic(sensitive_topic)

    # Create the dictionary-like structure to make the pd.DF conversion easier
    def create_contents(self):
        self.contents['date']= self.date
        self.contents['semantic tag'] = self.sentiment
        self.contents['label'] = self.topic
        self.contents['sensitive topic'] = self.sensitive_topic
        self.contents['text']=self.text

class Fetcher:
    """
    The Fetcher object be responsible for the following tasks:
    - collect the data found in the html files
    - sort through all the files and collect only the needed data(date, and text data)
    - create Tg_message objects and assign them the attributes: text and date
    - respect message restriction provided by the user
    """

    def __init__(self,base_path):
        self.path = os.path.dirname(__file__)#Where is the Fetcher object called from
        self.data_path = os.path.join(base_path, "Data/") #searches where the Data is located
        self.texts = []
        self.dates = []
    def read_html(self)->list:
        """
        The following method allows the Fetcher object to open a html, and using bs4 to sort through the contents.
        The output of this method is a list of bs4 tag objects, particularly the "body" divs, which are the singular messages in the html file.
        """

        file_name_list = os.listdir(self.data_path) # reads the file names from the data folder
        html_files = [file for file in file_name_list if file.endswith(".html")] # creates a list with the html files

        if not html_files: # handle no html file found
            print("No html files found")

        for file in html_files: # iterate through each html file
            file_path = os.path.join(self.data_path,file)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')  # create a soup object

                body_divs = soup.find_all('div', class_='body')
                bs_messages = [div for div in body_divs if div['class'] == ['body']]  # filter body divs, the to-be messages

                return bs_messages
    def create_messages(self,bs_messages: list,restriction:int)->list:
        """
        The following method transforms bs4 tag objects into Tg_message objects. Returns a list of TgMessage objects.
        """
        message_list = []
        for message in bs_messages:
            if message.find('div', class_='text') != None:  # if it does not find any texts, then we skip it
                text = message.find('div', class_='text').get_text().strip()
                date = message.find('div', class_='pull_right date details').get('title')

                date = date[:10].replace(".","/")
                message = TgMessage(text, date)
                message_list.append(message)

        return message_list[:restriction] # add message restriction for better performance

class Analyser:
    """
    The Analyser object creates LLM inferences. The outputs are predictions for topic, sentiment, and sensitive topics for the text provided.
    """
    def __init__(self):
        self.sentiment_tokenizer = None
        self.sentiment_analysis_model = None
        self.topic_classifier_model = None
        self.sensitive_topic_tokenizer = None
        self.sensitive_topic_model = None

        # Load sensitive topic mapping
        project_root = os.getcwd()
        s_topic_path = os.path.join(project_root, "src", "id2topic.json")

        if not os.path.exists(s_topic_path):
            raise FileNotFoundError(f"Could not find {s_topic_path}. Make sure the file exists in the src directory.")

        with open(s_topic_path) as f:
            self.target_variables = json.load(f)

    # Loads the tokenizer and model for LLM analysis if they are not already loaded. This should spare required compute resources for the text analysis.
    def load_sentiment_model(self):
        if self.sentiment_tokenizer is None or self.sentiment_analysis_model is None:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")
            self.sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")
    def load_topic_model(self):
        if self.topic_classifier_model is None:
            self.topic_classifier_model = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    def load_sensitive_topic_model(self):
        if self.sensitive_topic_tokenizer is None or self.sensitive_topic_model is None:
            self.sensitive_topic_tokenizer = AutoTokenizer.from_pretrained("apanc/russian-sensitive-topics")
            self.sensitive_topic_model = AutoModelForSequenceClassification.from_pretrained("apanc/russian-sensitive-topics")

    # create LLM inference
    def sentiment_analysis(self, analysed_data: str) -> str:
        """
        Performs sentiment analysis on the provided text using a pre-trained model.
        This method predicts whether the sentiment of the text is Neutral, Positive, or Negative.

        Args:
            analysed_data (str): The text to be analysed.

        Returns:
            str: The predicted sentiment label (Neutral, Positive, or Negative).
        """
        labels = ["Neutral", "Positive", "Negative"]
        inputs = self.sentiment_tokenizer(analysed_data, padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.sentiment_analysis_model(**inputs)

        predicted_class = torch.argmax(outputs.logits).item()
        return labels[predicted_class]
    def classify_topic(self, analysed_data: str) -> str:
        """
        Classifies the topic of the provided text using zero-shot classification.

        This method uses a pre-trained model to determine the most relevant topic
        from a predefined list of possible topics. Note that this analysis can be
        computationally intensive and may significantly impact system performance.

        **It is not recommended to run this analysis locally on systems with limited resources.**
        Consider using a cloud-based environment or a system with sufficient computational power.

        Args:
            analysed_data (str): The text to be analysed.

        Returns:
            str: The predicted topic label.
        """
        possible_labels = ['Politics', 'Economy', 'Technology', 'Sport', 'Culture', 'Health', 'Entertainment', 'Science',
                           'Environment', 'World News', 'Local News']
        output = self.topic_classifier_model(analysed_data, possible_labels, multi_label=False)
        return output['labels'][0]
    def classify_sensitive_topic(self, analysed_data: str) -> str:
        """
        Detects whether the provided text belongs to a predefined sensitive topic.
        This method uses a pre-trained model to classify the text into one of the target variables.

        Args:
            analysed_data (str): The text to be analysed.

        Returns:
            str: The predicted sensitive topic label.
        """
        inputs = self.sensitive_topic_tokenizer(analysed_data, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.sensitive_topic_model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()
        return self.target_variables[str(predicted_class)]

    def clear_models(self):
        """Clear all loaded models from memory."""
        self.sentiment_tokenizer = None
        self.sentiment_analysis_model = None
        self.topic_classifier_model = None
        self.sensitive_topic_tokenizer = None
        self.sensitive_topic_model = None

# to-be implemented
class Filter:
    """This class edits the resulting dataframe for topic, sensitive topic and date filtration."""
    def __init__(self):
        self.topic_filter = ['Politics', 'Economy', 'Technology', 'Sports', 'Health', 'Entertainment', 'Science',
                        'Environment', 'World News', 'Local News']
        self.date_filter = None
    def add_date(self,date:str):
        self.date_filter = date


    def filter_data(self,data:pd.DataFrame,user_topic)->pd.DataFrame:
        filtered_by_topic = data[data["Label"].str.contains(user_topic,case=False)]
        return filtered_by_topic

class Displayer:
    """
    The Displayer generates the data in the output folder. Using the provided data, a pd.DataFrame, it can generate a
    tabular file(CSV) and visualise the data using matplotlib.
    """
    def __init__(self,data:pd.DataFrame):
        self.data = data
    # Generate csv file
    def create_csv(self,output_file:str,sep):
        self.data.to_csv(output_file, sep=sep, index=False)
        print(f"Analysis completed. Results saved to {output_file}")

    def extract_labels_from_output_csv(self, output_dir: str, file_name: str = "output.csv") -> list:
        """
        Reads the default output CSV file and extracts unique labels (topics) from the 'Label' column.

        Args:
            output_dir (str): Path to the output directory where the CSV file is saved.
            file_name (str): Name of the output CSV file. Defaults to "output.csv".

        Returns:
            list: A list of unique labels as strings.
        """
        try:
            # Construct the full path to the output file
            file_path = os.path.join(output_dir, file_name)

            # Load the CSV file
            data = pd.read_csv(file_path, delimiter='|')

            # Extract unique labels from the 'Label' column
            unique_labels = data['Label'].str.strip().unique()

            # Convert to a list of strings
            labels_list = list(unique_labels)
            return labels_list
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except KeyError:
            print("The provided file does not contain a 'Label' column.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    # Visualising the Data
    def create_general_timeline(self, data: pd.DataFrame) -> plt:
        """
            Create a timeline plot showing the dynamic sum of daily semantic tags.

            Parameters:
            - data (pd.DataFrame): A DataFrame containing 'Date' and 'Semantic Tag' columns.
            """
        semantic_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        data['Semantic Value'] = data['Semantic Tag'].map(semantic_map)


        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

        # Group by date and sum the semantic values for each day
        daily_semantic_sum = data.groupby('Date')['Semantic Value'].sum().reset_index() #use reset_index to convert the date into a column

        # Ensure the x-axis includes all dates, even if no data is available for some messages
        full_date_range = pd.date_range(start=daily_semantic_sum['Date'].min(),
                                        end=daily_semantic_sum['Date'].max())
        daily_semantic_sum = daily_semantic_sum.set_index('Date').reindex(full_date_range, fill_value=0).reset_index()
        daily_semantic_sum.columns = ['Date', 'Semantic Value']

        # Plot the data
        plt.figure(figsize=(15, 5)) #resolution of the plot
        plt.plot(daily_semantic_sum['Date'], daily_semantic_sum['Semantic Value'], marker='o')
        plt.title('Semantic Tag Timeline')
        plt.xlabel('Date')
        plt.ylabel('Total Semantic Tag')
        plt.grid() #create sick grid look for the graph
        plt.xticks(daily_semantic_sum['Date'], rotation=45) #customize labels on the x-axis
        plt.tight_layout() # squish the labels and title
        return plt
    def create_topic_dynamics_timeline(self,topic_list: list, data: pd.DataFrame) -> plt:
        """
        Creates a timeline showing the topic frequency dynamics, ensuring all dates are included.

        Args:
            data (pd.DataFrame): The dataset containing the 'Date' and 'Label' columns.
            topic_list (list): The list of topics to include, each assigned a numeric ID.

        Returns:
            plt: The matplotlib plot object.
        """
        try:
            # Ensure required columns exist
            if "Date" not in data.columns or "Label" not in data.columns:
                raise KeyError("The dataset must contain 'Date' and 'Label' columns.")

            # Assign numeric IDs to topics, so they can be displayed on the y-axis
            topic_map = {topic: idx + 1 for idx, topic in enumerate(topic_list)}
            data['Topic ID'] = data['Label'].map(topic_map)

            # Convert 'Date' to datetime format
            data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

            # Generate a full date range from the dataset
            full_date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max())

            # Plot the timeline
            plt.figure(figsize=(16, 10))
            for topic, topic_id in topic_map.items():
                topic_data = data[data['Topic ID'] == topic_id]
                plt.scatter(topic_data['Date'], topic_data['Topic ID'], label=topic, s=50)

            plt.xticks(full_date_range, rotation=45)
            plt.yticks(range(1, len(topic_list) + 1), topic_list)
            plt.title("Topic Timeline")
            plt.xlabel("Date")
            plt.ylabel("Topics")
            plt.grid(True)
            plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            return plt
        except KeyError as e:
            print(f"KeyError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    def create_general_hist(self, data: pd.DataFrame) -> plt:
        """
           Create a histogram showing the semantic tags frequency across all topics.

           Parameters:
           - data (pd.DataFrame): A DataFrame containing a 'Semantic Tag' column.
           """
        try:
            # Map semantic tags to numeric values (optional, not needed for counts)
            semantic_counts = data['Semantic Tag'].value_counts()

            # Plot the histogram
            plt.figure(figsize=(15, 5))
            plt.bar(semantic_counts.index, semantic_counts.values, color='skyblue', edgecolor='black')
            plt.title("Semantic Tag Histogram")
            plt.xlabel("Semantic Tag")
            plt.ylabel("Frequency")
            plt.xticks(['Negative', 'Neutral', 'Positive'])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            return plt

        except KeyError as k:
            print(f"KeyError: {k}. Ensure the column names are correct.")
        except Exception as e:
            print(f"An error occurred: {e}")
    def create_hist_by_topic(self, user_topic, data: pd.DataFrame)->plt:
        """Semantic tag histogram of a specific topic"""
        try:
            # Check if the user_topic exists in the dataset
            if user_topic in set(data["Label"]):
                filtered_data = data[data['Label'] == user_topic]

                # Map semantic tags to numeric values
                semantic_counts = filtered_data['Semantic Tag'].value_counts()

                # Plot the histogram
                plt.figure(figsize=(15, 5))
                plt.bar(semantic_counts.index, semantic_counts.values, color='skyblue', edgecolor='black')
                plt.title(f"Frequency of Semantic Tags for Topic: {user_topic}")
                plt.xlabel("Semantic Tag")
                plt.ylabel("Frequency")
                plt.xticks(['Negative', 'Neutral', 'Positive'])
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                return plt
            else:
                raise ValueError(f"Topic '{user_topic}' not found in the data.")
        except KeyError as k:
            print(f"KeyError: {k}. Ensure the column names are correct.")
    def create_topic_frequency_hist(self,topic_list: list, data: pd.DataFrame) -> plt:
        """
        Creates a histogram showing the frequency of topics (labels) in the provided data.

        Args:
            data (pd.DataFrame): The dataset containing the 'Label' column.
            topic_list (list): A list of topics to include in the histogram.

        Returns:
            plt: The matplotlib plot object.
        """
        try:
            # Ensure the 'Label' column exists in the data
            if "Label" not in data.columns:
                raise KeyError("The dataset does not contain a 'Label' column.")

            # Count the frequency of each topic in the dataset
            topic_counts = data['Label'].value_counts()

            # Filter for the provided topic list to ensure all topics are included
            filtered_counts = topic_counts.reindex(topic_list, fill_value=0)

            # Plot the histogram
            plt.figure(figsize=(10, 6))
            plt.bar(filtered_counts.index, filtered_counts.values, color='skyblue', edgecolor='black')
            plt.title("Topic Frequency")
            plt.xlabel("Topic")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            return plt
        except KeyError as e:
            print(f"KeyError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    def create_timeline_by_topic(self, user_topic, data: pd.DataFrame)->plt:
        """Semantic tag timeline of a specific topic"""
        try:
            if user_topic in data["Label"].unique():# Check if the user_topic exists in the dataset
                filtered_data = data[data['Label'] == user_topic] # Filter the data for the selected topic
                filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], format='%d/%m/%Y')

                # Map semantic tags to numeric values
                semantic_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
                filtered_data['Semantic Tag Encoded'] = filtered_data['Semantic Tag'].map(semantic_map)

                # Group by date and calculate the sum
                grouped_data = filtered_data.groupby('Date')['Semantic Tag Encoded'].sum().reset_index()

                # Plot the timeline
                plt.figure(figsize=(15, 5))
                plt.plot(grouped_data['Date'], grouped_data['Semantic Tag Encoded'], marker='o', linestyle='-')
                plt.title(f"Semantic Tag Timeline for Topic: {user_topic}")
                plt.xlabel("Date")
                plt.ylabel("Semantic Tags Sum")
                plt.grid(True)
                plt.tight_layout()
                return plt
            else:
                print(f"Topic '{user_topic}' not found in the data. Please try another topic.")
        except KeyError as e:
            print(f"KeyError: {e}. Ensure the column names in the dataset are correct.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_plt(self,output_file:str, plt:plt):
        plt.savefigure(output_file)


# """"Testing the classes"""
#
# """Testing the Fetcher"""
# fetcher = Fetcher()
# bs_messages = fetcher.read_html()
# message_list = fetcher.create_messages(bs_messages)
# # print(text)
#
# """Testing the Analyser"""
# analyser = Analyser()
# # print(analyser.classify_topic(text))
# # print(analyser.sentiment_analysis(text))
# # print(analyser.classify_sensitive_topic(text))
#
# """Add new information to the Tg_Message object"""
# for message in message_list:
#     topic = analyser.classify_topic(message.text)
#     sentiment = analyser.sentiment_analysis(message.text)
#     sensitive_topic = analyser.classify_sensitive_topic(message.text)
#
#     message.assign_new_labels(topic,sentiment,sensitive_topic)
# num = 2
# text = f"{message_list[num].date},{message_list[num].topic},{message_list[num].sensitive_topic}"
# print(text)
# message_list[num].create_contents()
# message_dict = message_list[num].contents
# print(message_dict)