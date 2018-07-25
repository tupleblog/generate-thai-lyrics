# coding: utf-8
import os
import json
import torch
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split


class LyricCorpus(object):
    def __init__(self, file_path, n_filter=3):
        """
        Corpus class for Thai songs lyrics

        file_path: path to downloaded lyric, csv
        n_filter: filter out words from dictionary that appear less than n_filter times 
            across documents
        """
        lyric_df = pd.read_csv(file_path)
        lyrics = list(lyric_df.full_lyrics.dropna().map(self.clean_lyrics))
        train_text_list, val_text_list = train_test_split(lyrics, test_size=0.2)
        self.train_ = self.tokenize(train_text_list)
        self.valid_ = self.tokenize(val_text_list)
        self.dictionary = self.create_dictionary(n_filter=n_filter)
        self.dictionary_reverse = {v: k for k, v in self.dictionary.items()}
        self.train = self.word2idx(self.train_)
        self.valid = self.word2idx(self.valid_)

    def clean_lyrics(self, lyric):
        """
        Clean lines in lyric
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
                'ทำนอง / เรียบเรียง :'
            ]
            if any(line.startswith(s) for s in headers):
                pass
            else:
                line = ' '.join(line.replace('(', ' ').replace(')', ' ').replace('-', ' ').split())
                lyrics_clean.append(line)
        return '\n'.join(lyrics_clean).strip()

    def tokenize(self, text_list):
        """
        Tokenize Thai lyrics using deepcut
        """
        import deepcut
        words = []
        for lyric in tqdm(text_list):
            words.extend(deepcut.tokenize(lyric))
        return words

    def create_dictionary(self, n_filter=3):
        """
        Create dictionary from list of training and validation words

        Parameters
        ==========
        n_filter: filter out words from dictionary that appear less than n_filter times 
            across documents
        """
        words_dictionary = [k for k, v in Counter(self.train_ + self.valid_).items() if v >= n_filter]
        dictionary = {}
        dictionary['@@UNKNOWN@@'] = 0
        dictionary = {v: k for k, v in enumerate(list(words_dictionary), start=1)}
        return dictionary
    
    def word2idx(self, words):
        """
        Convert list of words into index of torch tensor
        """
        ids = np.array([self.dictionary.get(word, 0) for word in words])
        return torch.from_numpy(ids)