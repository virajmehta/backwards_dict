import multiprocessing
import requests
import string
from bs4 import BeautifulSoup as bs
from time import sleep
from string import ascii_lowercase


raw_data_file = 'raw_xword'


def parse_one_letter(letter, db):
    page = None
    if db == 'crosswordsolver.org':
        root_url = 'http://www.crosswordsolver.org'
        url = 'http://www.crosswordsolver.org/clues/' + letter
        page = requests.get(url)
        if page.status_code != 200:
            raise RuntimeWarning('letter {} not parsed'.format(letter))
    else:
        raise ValueError('db not supported')
    soup = bs(page.text,'lxml')
    container = soup.find(class_='container body')
    link_tags = container.find_all('a')
    urls = []
    for link in link_tags:
        urls.append(root_url + link['href'])

    pool = multiprocessing.Pool(processes=16)
    examples = pool.map(parse_one_page, urls)
    split_examples = []
    for clue, words in examples:
        if clue is None or words is None:
            continue
        for word in words:
            split_examples.append((clue, word))
    store_clue_words(split_examples, letter)


    # soup.find('strong').string to get the clue from the page
    # soup.find_all('div', class_='word') returns all the words for a clue
    # for each elt in the list, elt.string gives the word

'''takes examples, a list of (clue, word) tuples'''
def store_clue_words(examples, letter):
    f = open(raw_data_file +'_'+  letter, 'w')
    for clue, word in examples:
        f.write(clue.lower().strip() + ';' + word.lower().strip() + '\n')
    f.close()


def parse_one_page(url):
    x = True
    y = False
    while x:
        try:
            page = requests.get(url)
            x = False
        except:
            print 'they\'re getting mad!'
            sleep(10)
            x = True
            y = True
    if page.status_code != 200:
        print 'uh oh'
        return None, None
    if y:
        print 'phew'

    soup= bs(page.text, 'lxml')
    try:
        clue = str(soup.find('strong').string)
        words = [str(elt.string) for elt in soup.find_all('div', class_='word')]
        clue = clue.translate(None, string.punctuation)
    except:
        return (None, None)
    return (clue, words)

def main():
    for c in ascii_lowercase:
        print 'scraping letter {}'.format(c)
        parse_one_letter(c, 'crosswordsolver.org')
        print '{} complete'.format(c)


if __name__=='__main__':
    main()
