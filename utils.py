from itertools import chain
from collections import Counter
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
    soup = BeautifulSoup(requests.get('https://www.siamzone.com/music/thailyric/%d' % d).content, 'html.parser')
    title, artist_name = soup.find('title').text.split('|')
    title, artist_name = title.strip(), artist_name.strip()
    n_shares = int(soup.find('span', attrs={'class': 'sz-social-number'}).text.replace(',', ''))
    full_lyrics = soup.find('div', attrs={'itemprop': 'articleBody'}).text.strip()
    return {
        'url': 'https://www.siamzone.com/music/thailyric/%d' % d,
        'soup': soup, 
        'title': title,
        'artist_name': artist_name,
        'n_shares': n_shares,
        'full_lyrics': full_lyrics
    }

def scrape_siamzon():
    scraped_siamzone = []
    for i in range(14050, 16041):
        try:
            scraped_siamzone.append(scrape_siamzone_url(i))
        except:
            pass

    scraped_siamzone_df = pd.DataFrame(scraped_siamzone)
    scraped_siamzone_df['lyrics'] = scraped_siamzone_df.full_lyrics.map(clean_lyrics)
    return scraped_siamzone_df