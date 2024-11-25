import os
from datetime import date

from Tools.demo.eiffel import Tests
from bs4 import BeautifulSoup


class Tg_Message:
    """this class will create objects that reseamble telegram messages, but with a dictionary like structure.
    A sort of container.
    It will have the following attributes:
    self.text = text
    self.date = date
    self.contents = {
                    "text":self.text,
                    "date":self.date
    }"""

    def __init__(self,text:str,date):
        self.text = text
        self.date = date
        self.contents={}
class Fetcher:
    """
    The Fetcher wil be responsible for the following tasks:
    - collect the data found in the html files
    - sort through the all the files and collect only the needed data
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

                return bs_messages # messages is a list of bs4 tag objects
    def create_messages(self,bs_messages: list):
        """
        The following method transforms bs4 tag objects Tg_message objects.
        """
        message_list = []
        for message in bs_messages:
            if message.find('div', class_='text') != None:  # if it does not find any texts, then we skip it
                text = message.find('div', class_='text').get_text().strip()
                date = message.find('div', class_='pull_right date details').get('title')
                message = Tg_Message(text, date)
                message_list.append(message)

        return message_list
class Analyser:
    """
    This class will:
    a) pre-processing of the collected data by the fetcher
    b) semantic analysis of the text-date within the messages
    """
class Displayer:
    """
    The Displayer will create the csv file.
    """


""""Testing the classes"""
fetcher = Fetcher()
bs_messages = fetcher.read_html()
message_list = fetcher.create_messages(bs_messages)
print(message_list[0].text)
