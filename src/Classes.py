import os
from bs4 import BeautifulSoup
from transformers import pipeline,AutoTokenizer, AutoModelForSequenceClassification
import torch, json


class Tg_Message():
    """this class will create objects resembling telegram messages, but with a dictionary-like structure.
    A sort of container.
    It will have the following attributes:
    self.text = text
    self.date = date
    self.contents = {
                    "text":self.text,
                    "date":self.date
    }
    @to-do
    It will be a good idea to create a method, which will assign these new attributes to the Tg_Message object.
    add:
    self.topic = topic
    self.sentiment = sentiment
    self.sensitive_topic = sensitive_topic
    """

    def __init__(self,text:str,date):
        self.text = text
        self.date = date
        self.contents={}

    def assign_topic(self,topic:str): # wil it be a good idea to create just one method for them all?
        self.topic = topic
    def assign_sentiment(self,sentiment:str):
        self.sentiment = sentiment
    def assign_sensitive_topic(self,sensitive_topic:str):
        self.sensitive_topic = sensitive_topic

    def assign_new_labels(self,topic:str,sentiment:str,sensitive_topic:str):
        self.assign_topic(topic)
        self.assign_sentiment(sentiment)
        self.assign_sensitive_topic(sensitive_topic)

    def create_contents(self):
        self.contents['date']= self.date
        self.contents['semantic tag'] = self.sentiment
        self.contents['label'] = self.topic
        self.contents['sensitive topic'] = self.sensitive_topic
        self.contents['text']=self.text
class Fetcher:
    """
    The Fetcher wil be responsible for the following tasks:
    - collect the data found in the html files
    - sort through the all the files and collect only the needed data(date, and text data)
    - create Tg_message objects and assign them the attributes: text and date
    @to-do
    - output a dictionary with the following structure:
        {int: { "date"           :date.object,
                "text_contents" :str }}
    """
    def __init__(self):
        self.path = os.path.dirname(__file__)#Where is the Fetcher object called from
        self.data_path = os.path.join(self.path, "Data/") #searches where the Data is located
        self.texts = []
        self.dates = []
    def read_html(self)->list:
        """
        The following method allows the Fetcher object to open a html, and using bs4 to sort through the contents.
        It a list of bs4 tag objects, particularly the "body" divs.
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
    def create_messages(self,bs_messages: list):
        """
        The following method transforms bs4 tag objects Tg_message objects.
        """
        message_list = []
        for message in bs_messages:
            if message.find('div', class_='text') != None:  # if it does not find any texts, then we skip it
                text = message.find('div', class_='text').get_text().strip()
                date = message.find('div', class_='pull_right date details').get('title')

                date = date[:10].replace(".","/")
                message = Tg_Message(text, date)
                message_list.append(message)

        return message_list[:25] # add message restriction for better performance
class Analyser:
    """
    The Analyser class boots up the LLMs, and provides the prediction for topic, sentiment, and sensitive topic for the
    text provided.
    """
    def __init__(self):
        #Initializing model for sentiment analysis
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")
        self.sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")

        #Initialiaze model for zero-shot classification
        self.topic_classifier_model = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

        # Initialize model for sensitive topic classification
        self.sensitive_topic_tokenizer = AutoTokenizer.from_pretrained("apanc/russian-sensitive-topics")
        self.sensitive_topic_model = AutoModelForSequenceClassification.from_pretrained("apanc/russian-sensitive-topics")
        with open("src/id2topic.json") as f:
            self.target_variables = json.load(f)

        # self.inappropirate_messages
    def sentiment_analysis(self,analysed_data:str) -> str:
        labels = ["Neutral", "Positive", "Negative"]
        inputs = self.sentiment_tokenizer(analysed_data, padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.sentiment_analysis_model(**inputs)

        # Extract predicted sentiment
        predicted_class = torch.argmax(outputs.logits).item()
        sentiment = labels[predicted_class]
        return sentiment
    def classify_topic(self,analysed_data:str):
        possible_labels = ['Politics', 'Economy', 'Technology', 'Sports', 'Health', 'Entertainment', 'Science',
                        'Environment', 'World News', 'Local News']
        output = self.topic_classifier_model(analysed_data, possible_labels, multi_label=False)
        return output['labels'][0]
    def classify_sensitive_topic(self,analysed_data:str):
        inputs = self.sensitive_topic_tokenizer(analysed_data, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.sensitive_topic_model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()
        predicted_sensitive_topic = self.target_variables[str(predicted_class)]
        return predicted_sensitive_topic
class Displayer:
    """
    The Displayer will create the csv file.
    """


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