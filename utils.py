from itertools import chain
from collections import Counter
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd


def flatten(ls):
    """
    Flatten list of list
    """
    return list(chain.from_iterable(ls))

def clean_lyrics(lyric):
    """
    Clean lines that do not contain lyrics
    """
    lines = lyric.split('\n')
    lyrics_clean = [] 
    for line in lines:
        # remove headers from the file
        headers = [
            'เพลง ', 'คำร้อง ', 'คำร้อง/ทำนอง ', 'ศิลปิน ', 'ทำนอง ', 
            'เรียบเรียง ', 'เพลงประกอบละคร ', 'อัลบัม ', 'ร่วมร้องโดย ', 
            'เนื้อร้อง/ทำนอง', 'ทำนอง/เรียบเรียง ', 'เพลงประกอบภาพยนตร์ ', 
            'เพลงประกอบละครซิทคอม ', 'คำร้อง/ทำนอง/เรียบเรียง ', 
            'คำร้อง/เรียบเรียง ', 'เพลงประกอบ ', 'ร้องโดย ', 
            'ทำนอง / เรียบเรียง :', ' สังกัด'
        ]
        if any(line.startswith(s) for s in headers):
            pass
        else:
            line = ' '.join(line.replace('(', ' ').replace(')', ' ').replace('-', ' ').split())
            lyrics_clean.append(line)
    return '\n'.join(lyrics_clean).strip()


def create_lookup_dict(tokenized_lyrics, n_min=None):
    """
    Create lookup dictionary from list of words (lyrics)
    """
    word_counts = Counter(tokenized_lyrics)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    if n_min is not None:
        sorted_vocab = {k: v for k, v in word_counts.items() if v >= n_min}
    vocab_to_int = {word: i for i, word in enumerate(sorted_vocab, 0)}
    int_to_vocab = {i: word for word, i in vocab_to_int.items()}
    return (vocab_to_int, int_to_vocab)


def scrape_siamzone_url(d):
    """
    Script to scrape Siamzone lyrics from a given song_id (integer)
    """
    soup = BeautifulSoup(requests.get('https://www.siamzone.com/music/thailyric/{}'.format(d)).content, 'html.parser')
    song_title, artist_name = soup.find('title').text.split('|')
    song_title, artist_name = song_title.replace("เนื้อเพลง ", "").strip(), artist_name.strip()
    try:
        n_views = ' '.join(soup.find('div', attrs={'class': 'has-text-info'}).text.strip().split())
    except:
        n_views = ''
    try:
        full_lyrics = soup.find_all('div', attrs={'class': 'column is-6-desktop'})[1]
        lyrics = full_lyrics.find("div", attrs={'style': "margin-bottom: 1rem;"}).text.strip()
    except:
        lyrics = ""
    return {
        'url': 'https://www.siamzone.com/music/thailyric/%d' % d,
        'soup': soup, 
        'song_title': song_title,
        'artist_name': artist_name,
        'n_views': n_views,
        'lyrics': lyrics
    }


def scrape_siamzone(start=1, end=20200):
    """
    Scrape Siamzon URL and return dictioanry output

    Usage
    =====
    >>> scraped_siamzone_df = scrape_siamzone(start=1, end=20200)
    >>> scraped_siamzone_df['html'] = scraped_siamzone_df.soup.map(lambda x: x.prettify())
    """
    scraped_siamzone = []
    for i in tqdm(range(start, end)):
        try:
            scraped_siamzone.append(scrape_siamzone_url(i))
        except:
            pass
    scraped_siamzone_df = pd.DataFrame(scraped_siamzone)
    return scraped_siamzone_df
