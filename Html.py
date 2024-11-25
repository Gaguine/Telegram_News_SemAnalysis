# import os, requests
from distutils.command.clean import clean
import re
from bs4 import BeautifulSoup
list_dates = []
list_texts = []

# script_dir = os.path.dirname(__file__)
file_path = 'Data/messages.html'
# response = requests.get('file://'+file_path)
# html_content = response.text

with open(file_path,'r',encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser') # create a soup object

    body_divs = soup.find_all('div',class_='body')
    messages = [div for div in body_divs if div['class']==['body']] # filter body divs, the messages

    for message in messages:
        if message.find('div',class_='text') != None: # if it does not find any texts, then we skip it
            list_texts.append(message.find('div',class_='text').get_text().strip())
            list_dates.append(message.find('div',class_='pull_right date details').get('title'))
#
# print(len(list_dates),len(list_texts))
# print(list_texts[0],list_dates[0])
# print(list_texts[1],list_dates[1])
# print(list_texts[50],list_dates[50])
# print(list_texts[-1],list_dates[-1])


