# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import logging
import re

import torch
import torch.nn.functional as F

from src.transformers import BertTokenizer, BertForMaskedLM
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


import string
from nltk.corpus import words

ALPHA = string.ascii_letters

logger = logging.getLogger(__name__)
import math


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class Cmasked:
    def __init__(self, max_seq_length, do_lower_case, pre_trained="bert-large-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BertForMaskedLM.from_pretrained(pre_trained)
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained, do_lower_case=do_lower_case)

        self.model.to(self.device)
        self.max_seq_length = max_seq_length

        self.model.eval()

        self.lemmatizer = WordNetLemmatizer()
        self.limit_bert = 512

    def clean(self, text_a_raw, masked_id):
        pat_is = re.compile("(it|he|she|that|this|there|here) \'s", re.I)
        # to find the 's following the letters

        # to find the abbreviation of not
        pat_not = re.compile("(?<=[a-zA-Z]) n\'t")
        # to find the abbreviation of would
        pat_would = re.compile("(?<=[a-zA-Z]) \'d")
        # to find the abbreviation of will
        pat_will = re.compile("(?<=[a-zA-Z]) \'ll")
        # to find the abbreviation of am
        pat_am = re.compile("(?<=[I|i]) \'m")
        # to find the abbreviation of are
        pat_are = re.compile("(?<=[a-zA-Z]) \'re")
        # to find the abbreviation of have
        pat_ve = re.compile("(?<=[a-zA-Z]) \'ve")
        new_text = pat_is.sub(r"\1 is", text_a_raw)

        new_text = pat_not.sub(" not", new_text)
        new_text = pat_would.sub(" would", new_text)
        new_text = pat_will.sub(" will", new_text)
        new_text = pat_am.sub(" am", new_text)
        new_text = pat_are.sub(" are", new_text)
        text_a_raw = pat_ve.sub(" have", new_text)

        text_a_raw = text_a_raw.split(' ')

        temp_index = 0
        while '' in text_a_raw:
            empty_index = text_a_raw.index('')
            text_a_raw.remove('')
            if empty_index < masked_id - temp_index:
                temp_index = temp_index + 1

        return text_a_raw, temp_index

    def clean_word(self, word):
        pat_is = re.compile("(it|he|she|that|this|there|here) \'s", re.I)
        # to find the 's following the letters

        # to find the abbreviation of not
        pat_not = re.compile("(?<=[a-zA-Z]) n\'t")
        # to find the abbreviation of would
        pat_would = re.compile("(?<=[a-zA-Z]) \'d")
        # to find the abbreviation of will
        pat_will = re.compile("(?<=[a-zA-Z]) \'ll")
        # to find the abbreviation of am
        pat_am = re.compile("(?<=[I|i]) \'m")
        # to find the abbreviation of are
        pat_are = re.compile("(?<=[a-zA-Z]) \'re")
        # to find the abbreviation of have
        pat_ve = re.compile("(?<=[a-zA-Z]) \'ve")
        new_text = pat_is.sub(r"\1 is", word)

        new_text = pat_not.sub(" not", new_text)
        new_text = pat_would.sub(" would", new_text)
        new_text = pat_will.sub(" will", new_text)
        new_text = pat_am.sub(" am", new_text)
        new_text = pat_are.sub(" are", new_text)
        text_a_raw = pat_ve.sub(" have", new_text)
        text_a_raw = text_a_raw.split(' ')
        while '' in text_a_raw:
            text_a_raw.remove('')
        text_a_raw = " ".join(text_a_raw)
        return text_a_raw

    def clean_word_proposed(self, word):
        pat_is = re.compile("(it|he|she|that|this|there|here) \'s", re.I)
        # to find the 's following the letters
        pat_s = re.compile("(?<=[a-zA-Z])\'s")
        # to find the ' following the words ending by s
        pat_s2 = re.compile("(?<=s)\'s?")
        # to find the abbreviation of not
        pat_not = re.compile("(?<=[a-zA-Z]) n\'t")
        # to find the abbreviation of would
        pat_would = re.compile("(?<=[a-zA-Z]) \'d")
        # to find the abbreviation of will
        pat_will = re.compile("(?<=[a-zA-Z]) \'ll")
        # to find the abbreviation of am
        pat_am = re.compile("(?<=[I|i]) \'m")
        # to find the abbreviation of are
        pat_are = re.compile("(?<=[a-zA-Z]) \'re")
        # to find the abbreviation of have
        pat_ve = re.compile("(?<=[a-zA-Z]) \'ve")
        new_text = pat_is.sub(r"\1 is", word)
        new_text = pat_s.sub("", new_text)
        new_text = pat_s2.sub("", new_text)
        new_text = pat_not.sub(" not", new_text)
        new_text = pat_would.sub(" would", new_text)
        new_text = pat_will.sub(" will", new_text)
        new_text = pat_am.sub(" am", new_text)
        new_text = pat_are.sub(" are", new_text)
        text_a_raw = pat_ve.sub(" have", new_text)
        text_a_raw = text_a_raw.replace('\'', ' ')
        text_a_raw = text_a_raw.split(' ')
        while '' in text_a_raw:
            text_a_raw.remove('')
        text_a_raw = " ".join(text_a_raw)
        return text_a_raw

    def pre_processed_text(self, line, masked_id, noise_type):
        self.target_start_id = None

        masked_id = int(masked_id)
        text_a_raw = line.split(' ')  # text_a untokenized
        if noise_type == "MASKED":
            text_a_raw[masked_id] = "[MASK]"

        masked_word = text_a_raw[masked_id]  # get the target word
        text_a_raw = ' '.join(text_a_raw)

        text_a_raw, temp_index = self.clean(text_a_raw, masked_id)

        masked_word = self.clean_word(masked_word)
        masked_id = text_a_raw.index(masked_word, masked_id - temp_index)

        original_text = ' '.join(text_a_raw)
        sub_tokens = self.find_tokens(text_a_raw[:masked_id])

        features = self.tokenizer.encode_plus(
            original_text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        text_a_tokenized = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0])
        masked_word_tokenized = self.tokenizer.tokenize(masked_word)  # tokenize target word

        target_word_start_index = masked_id + sub_tokens
        if self.target_start_id is not None:
            if target_word_start_index != self.target_start_id:
                print(self.target_word)
                exit(5)
        target_word_end_index = target_word_start_index + len(masked_word_tokenized) - 1

        text = ' '.join(text_a_tokenized)
        word_index = target_word_start_index

        return text, word_index, target_word_end_index, features

    def find_tokens(self, text_list):

        sub_tokens = 1  # clk token
        for word in text_list:
            token_list = self.tokenizer.tokenize(word)
            sub_tokens = sub_tokens + len(token_list) - 1

        return sub_tokens

    def pre_processed_text_temp(self, line, masked_id, noise_type):
        self.target_start_id = None

        masked_id = int(masked_id)
        text_a_raw = line.split(' ')  # text_a untokenized

        masked_word = text_a_raw[masked_id]  # get the target word
        text_a_raw = ' '.join(text_a_raw)

        text_a_raw, temp_index = self.clean(text_a_raw, masked_id)

        masked_word = self.clean_word(masked_word)
        masked_id = text_a_raw.index(masked_word, masked_id - temp_index)
        sub_tokens = self.find_tokens(text_a_raw[:masked_id])

        original_text = ' '.join(text_a_raw)
        features = self.tokenizer.encode_plus(
            original_text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        text_a_tokenized = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0])
        masked_word_tokenized = self.tokenizer.tokenize(masked_word)  # tokenize target word

        target_word_start_index = masked_id + sub_tokens
        if self.target_start_id is not None:
            if target_word_start_index != self.target_start_id:
                print(self.target_word)
                exit(5)
        target_word_end_index = target_word_start_index + len(masked_word_tokenized) - 1

        text = ' '.join(text_a_tokenized)
        word_index = target_word_start_index

        return text, word_index, target_word_end_index, features

    def proposed_candidates(self, sentences, word, word_id, noise_type, synonyms=[], top_k=30,
                            proposed_words_temp=None):
        proposed_words = {}
        text, target_word_start_index, target_word_end_index, features = self.pre_processed_text(sentences, word_id,
                                                                                                 noise_type)

        if noise_type == "MASKED":
            text_temp, target_word_start_index_temp, target_word_end_index_temp, features_temp = self.pre_processed_text_temp(
                sentences, word_id,
                noise_type)

        masked_id = target_word_start_index

        if target_word_start_index == target_word_end_index and noise_type != "MASKED":
            vocab_id = self.tokenizer.convert_tokens_to_ids(text.split(" ")[target_word_start_index])
        else:
            vocab_id = -1
        input_ids = features['input_ids']
        input_mask = features['attention_mask']
        segment_ids = features['token_type_ids']

        self.input_mask = input_mask
        self.segment_ids = segment_ids

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        synonyms_id = []
        for word in synonyms:
            token_list = self.tokenizer.tokenize(word)
            if len(token_list) == 1 and token_list[0] != '[UNK]':
                synonyms_id.append(self.tokenizer.convert_tokens_to_ids(token_list))

        if len(synonyms_id) == 0:
            synonyms_id = None

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                noise_type=noise_type, word_index=masked_id, input_ids_synonyms=synonyms_id)

        possible_index = self.possible_index[:]
        # not the same word
        try:
            if noise_type == "MASKED":
                possible_index.remove(
                    self.tokenizer.convert_tokens_to_ids(text_temp.split(" ")[target_word_start_index]))
            else:
                if vocab_id != -1:
                    possible_index.remove(vocab_id)
        except:
            pass

        output_prediction = output[0][0][masked_id][possible_index]
        output_prediction = F.softmax(output_prediction, dim=-1)
        top_k_words_index = torch.topk(output_prediction, top_k)[1].detach().cpu().numpy()
        if proposed_words_temp is not None:
            for word in proposed_words_temp:
                proposed_words[word] = 0
        i = 0
        lenght_dict = len(proposed_words)
        while lenght_dict < top_k:
            proposed_words[self.tokenizer.convert_ids_to_tokens(possible_index[top_k_words_index[i]])] = 0
            lenght_dict = len(proposed_words)

            i = i + 1

        return proposed_words

    def get_index(self, list_candidates, candidate):
        if candidate in list_candidates:
            return list_candidates.index(candidate)
        else:
            return 200

    def proposed_candidates_train(self, sentences, word, word_id, noise_type, synonyms=[], top_k=50):
        proposed_words = {}
        text, target_word_start_index, target_word_end_index, features = self.pre_processed_text(sentences, word_id,
                                                                                                 noise_type)
        masked_id = target_word_start_index

        if target_word_start_index == target_word_end_index and noise_type != "MASKED":
            vocab_id = self.tokenizer.convert_tokens_to_ids(word)
        else:
            vocab_id = -1
        input_ids = features['input_ids']
        input_mask = features['attention_mask']
        segment_ids = features['token_type_ids']

        self.input_mask = input_mask
        self.segment_ids = segment_ids

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        synonyms_id = []
        for word in synonyms:
            token_list = self.tokenizer.tokenize(word)
            if len(token_list) == 1 and token_list[0] != '[UNK]':
                synonyms_id.append(self.tokenizer.convert_tokens_to_ids(token_list[0]))

        if len(synonyms_id) == 0:
            synonyms_id = None

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                noise_type=noise_type, word_index=masked_id, input_ids_synonyms=synonyms_id)

        possible_index = self.possible_index[:]
        # not the same word
        if vocab_id != -1:
            possible_index.remove(vocab_id)
        output_prediction = output[0][0][masked_id][possible_index]
        output_prediction = F.softmax(output_prediction, dim=-1)
        top_k_words_index = torch.topk(output_prediction, top_k)[1].detach().cpu().numpy()
        top_k_words_prop = torch.topk(output_prediction, top_k)[0].detach().cpu().numpy()
        for i in range(0, len(top_k_words_prop)):
            proposed_words[self.tokenizer.convert_ids_to_tokens(possible_index[top_k_words_index[i]])] = []

        return proposed_words

    def predictions(self, sentences, word, main_word, word_id, proposed_words, noise_type, synonyms=[]):

        proposed_words_temp = {}
        text, target_word_start_index, target_word_end_index, features = self.pre_processed_text(sentences, word_id,
                                                                                                 noise_type)

        if noise_type == "MASKED":
            text_temp, target_word_start_index_temp, target_word_end_index_temp, features_temp = self.pre_processed_text_temp(
                sentences, word_id,
                noise_type)

        masked_id = target_word_start_index

        input_ids = features['input_ids']
        input_mask = features['attention_mask']
        segment_ids = features['token_type_ids']

        self.input_mask = input_mask
        self.segment_ids = segment_ids

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        synonyms_id = []

        for word in synonyms:
            token_list = self.tokenizer.tokenize(word)
            if len(token_list) == 1 and token_list[0] != '[UNK]':
                synonyms_id.append(self.tokenizer.convert_tokens_to_ids(token_list))

        if len(synonyms_id) == 0:
            synonyms_id = None
        else:
            pass
            # synonyms_id = torch.tensor(synonyms_id)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                noise_type=noise_type, word_index=masked_id, input_ids_synonyms=synonyms_id)

        possible_index, multiple_word = self.get_index_from_possible_words(proposed_words)

        try:
            if noise_type == "MASKED":
                possible_index.remove(
                    self.tokenizer.convert_tokens_to_ids(text_temp.split(" ")[target_word_start_index]))

            else:
                possible_index.remove(input_ids[0][masked_id].item())
        except:
            pass

        output_prediction = output[0][0][masked_id][possible_index]


        output_prediction = F.softmax(output_prediction, dim=-1)

        topk = torch.topk(output_prediction, len(possible_index))
        top_k_words_index = topk.indices.detach().cpu().numpy()
        top_k_words_prop = topk.values.detach().cpu().numpy()

        if main_word.split('.')[0] == "":
            word_temp = "."
        else:
            word_temp = main_word.split('.')[0]
        self.target_word = word_temp
        target_pos = main_word.split('.')[-1]
        to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
        from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}
        try:
            self.pos_initial = to_wordnet_pos[from_lst_pos[target_pos]]
        except:
            self.pos_initial = to_wordnet_pos[target_pos]

        for i in range(0, len(possible_index)):
            candidate_word = self.clean_to_proposal[self.tokenizer.convert_ids_to_tokens(possible_index[i])]
            if candidate_word == self.target_word or \
                    self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                    self.target_word in candidate_word:
                proposed_words_temp[candidate_word] = 0
            else:
                proposed_words_temp[candidate_word] = output_prediction[i].item()


        if noise_type == "MASKED":
            proposed_words_temp[self.tokenizer.convert_tokens_to_ids(text_temp.split(" ")[target_word_start_index])] = 0
        else:
            proposed_words_temp[self.tokenizer.convert_ids_to_tokens(input_ids[0][masked_id].item())] = 0

        for word in proposed_words:
            if word not in proposed_words_temp:
                proposed_words_temp[word] = 0


        return proposed_words_temp


    def get_index_from_possible_words(self, proposed_words):
        val = 100
        possible_index = []
        self.clean_to_proposal = {}
        multiple_index = {}

        for word in proposed_words:

            index = self.tokenizer.convert_tokens_to_ids(word)
            if index == val:
                word_temp = self.clean_word_proposed(word)
                index_temp = self.tokenizer.convert_tokens_to_ids(word_temp)
                if index_temp != 100:
                    self.clean_to_proposal[word_temp] = word
                    possible_index.append(index_temp)
                else:
                    multiple_index[word] = 0

            else:
                self.clean_to_proposal[word] = word
                possible_index.append(index)

        return possible_index, multiple_index

    def get_possible_words(self):
        word_set = set(words.words())
        self.possible_index = []
        for name in self.tokenizer.get_vocab():
            if name.startswith(tuple(ALPHA)) and len(name) > 1 and name in word_set:
                name_index = self.tokenizer.convert_tokens_to_ids(name)
                self.possible_index.append(name_index)

        return

    def get_proposed_words(self, proposed_words, input_ids, input_mask, segment_ids, masked_id, noise_type, word_index,
                           ratio=1):

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                noise_type=noise_type, word_index=masked_id)

        val = 100
        possible_index = self.tokenizer.convert_tokens_to_ids(proposed_words)
        possible_index = list(filter(lambda x: x != val, possible_index))
        try:
            possible_index.remove(word_index)
        except:
            pass

        output_prediction = output[0][0][masked_id][possible_index]
        output_prediction = F.softmax(output_prediction, dim=-1)
        top_k_words_index = torch.topk(output_prediction, len(possible_index))[1].detach().cpu().numpy()
        top_k_words_prop = torch.topk(output_prediction, len(possible_index))[0].detach().cpu().numpy()
        for i in range(0, len(top_k_words_prop)):
            candidate_word = self.tokenizer.convert_ids_to_tokens(possible_index[top_k_words_index[i]])
            if candidate_word == self.target_word or \
                    self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                    self.target_word in candidate_word:
                proposed_words[candidate_word] = 0
            else:
                proposed_words[candidate_word] = proposed_words[candidate_word] + ratio * top_k_words_prop[i]
        return proposed_words

    def get_proposed_words_candidate(self, input_ids, input_mask, segment_ids, masked_id, noise_type, word_index,
                                     vocab_id=None,
                                     top_k=None, proposed_words=None):

        candidate = []
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                noise_type=noise_type, word_index=masked_id)

        if top_k is None:
            val = 100
            possible_index = self.tokenizer.convert_tokens_to_ids(proposed_words)
            possible_index = list(filter(lambda x: x != val, possible_index))
            try:
                possible_index.remove(word_index)
            except:
                pass
            top_k = len(possible_index)
        else:
            possible_index = self.possible_index[:]
            # not the same word
            if vocab_id != -1:
                try:
                    possible_index.remove(vocab_id)
                except:
                    pass

        output_prediction = output[0][0][masked_id][possible_index]
        output_prediction = F.softmax(output_prediction, dim=-1)

        top_k_words_index = torch.topk(output_prediction, top_k)[1].detach().cpu().numpy()
        top_k_words_prop = torch.topk(output_prediction, top_k)[0].detach().cpu().numpy()
        for i in range(0, len(top_k_words_prop)):
            candidate.append(self.tokenizer.convert_ids_to_tokens(possible_index[top_k_words_index[i]]))
        return candidate
