import os
from datetime import date
from bs4 import BeautifulSoup


"""
The Fetcher wil be responsible for the following tasks:
- collect the data found in the html files
- sort through the all the files and collect only the used data
- output a dictionary with the following structure:
    {int: { "date"           :date.object,
            "text_contents" :str }}

when reading be mindful of the tags, add each in a list it will be easy to store , create a dictionary

"""
class Fetcher:
    def __init__(self):
        self.path = os.path.dirname(__file__)#Where is the Fetcher object called from
        self.data_path = os.path.join(self.path, "Data/") #controlla se hai formulato in maniera coretta
        self.texts = []
        self.dates = []

    def read_html(self):

        file_name_list = os.listdir(self.data_path) # reads the file names
        html_files = [file for file in file_name_list if file.endswith(".html")] # create a list with the html files

        if not html_files: # handle no html file found
            print("No html files found")

        for file in html_files:
            file_path = os.path.join(self.data_path,file)
            with open(file_path,'r',encoding='utf-8') as content: # Be mindful that there may be many, many html files. Iterate.
                soup = BeautifulSoup(content, 'html.parser')

                for text in soup.find_all('div', class_="text"):
                    self.texts.append(text)

                for date in soup.find_all('div', class_="pull_right date details"):
                    self.dates.append(date.get('title'))
        print(self.texts[-1],self.dates[-1])
        print(self.texts[1],self.dates[0])
fetcher = Fetcher()
fetcher.read_html()