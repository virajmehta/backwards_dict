import requests
from bs4 import BeautifulSoup as bs
from string import ascii_lowercase

def parse_one_letter(letter, db):
    pass
    # soup.find('strong').string to get the clue from the page
    # soup.find_all('div', class_='word') returns all the words for a clue
    # for each elt in the list, elt.string gives the word

def parse_one_page(url):
    soup= bs(requests.get(url))
    clue = soup.find('strong').string
    words = [elt.string for elt in soup.find_all('div', class_='word')]
    return clue, words

def main():
    pass



if __name__=='__main__':
    main()
