# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import logging
import torch
import torch.nn.functional as F
import numpy as np


from transformers import BertForSequenceClassification, BertTokenizer


from sentence_transformers import SentenceTransformer, util
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import re

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class CGLOSS:
    def __init__(self, pre_training_model, max_seq_length, do_lower_case, num_labels=2,
                 model="stsb-roberta-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BertForSequenceClassification.from_pretrained(pre_training_model, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(pre_training_model, do_lower_case=do_lower_case)
        self.model.to(self.device)
        self.model.eval()

        self.model_sentence = SentenceTransformer(model)
        self.model_sentence.to(self.device)
        self.model_sentence.eval()

        self.max_seq_length = max_seq_length
        self.num_labels = num_labels

        self.main_gloss_embedding = None
        self.lemmatizer = WordNetLemmatizer()

    def update_sentence_bert(self, path):
        self.model_sentence = SentenceTransformer(path)
        self.model_sentence.to(self.device)
        self.model_sentence.eval()

    def clean(self, text_a_raw):
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

        while '' in text_a_raw:
            text_a_raw.remove('')

        return ' '.join(text_a_raw)

    def save_main_gloss_embedding(self, gloss_embedding):
        self.main_gloss_embedding = gloss_embedding

    def convert_examples_to_features(self, sentence, gloss_list, top_k, max_seq_length=512, clean_flag=True):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        if clean_flag:
            sentence = self.clean(sentence)

        for i in range(0, top_k):
            if i < len(gloss_list):
                gloss_sentence = gloss_list[i]
                if clean_flag:
                    gloss_sentence = self.clean(gloss_sentence)

                feature = self.tokenizer.encode_plus(
                    sentence,
                    gloss_sentence,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                    padding='max_length',
                    return_tensors='pt',
                    truncation=True
                )

                features.append(
                    InputFeatures(input_ids=feature['input_ids'].to(self.device),
                                  input_mask=feature['attention_mask'].to(self.device),
                                  segment_ids=feature['token_type_ids'].to(self.device)))
            else:
                break
        return features



    def find_best_gloss(self, original_sentences, gloss_list, word, lemmas=None, candidate=True, clean_flag=True, top_k=4):


        if candidate:
            candidate_word = word
            if candidate_word == self.target_word or \
                    self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                    self.target_word in candidate_word:
                substituability_score = 0
                return "_", substituability_score
        else:
            main_word = word
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

        if top_k is None:
            top_k = len(gloss_list)

        features = self.convert_examples_to_features(original_sentences, gloss_list, top_k, clean_flag=clean_flag)

        input_ids = torch.stack([f.input_ids for f in features]).squeeze(1)
        input_mask = torch.stack([f.input_mask for f in features]).squeeze(1)
        segment_ids = torch.stack([f.segment_ids for f in features]).squeeze(1)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)

        logits_ = F.softmax(logits[0], dim=-1)

        logits_ = logits_.detach().cpu().numpy()
        output = np.argmax(logits_, axis=0)[1]

        best_gloss = gloss_list[output]

        # if we use the weak supervisor signal we get the gloss after the :
        best_gloss = " ".join(best_gloss.split(":")[1:])
        if lemmas is not None:
            best_lemmas = lemmas[output]
            best_gloss = best_lemmas + " - " + best_gloss
        if clean_flag:
            best_gloss = self.clean(best_gloss)
        gloss_embedding = self.get_gloss_embedding(best_gloss)
        cosine = 0

        # what is used in the sentencebert Semantic Textual Similarity
        if self.main_gloss_embedding is not None:
            cosine = util.pytorch_cos_sim(self.main_gloss_embedding, gloss_embedding)[0][0].item()


        return gloss_embedding, cosine

    def find_best_gloss_sentence(self, original_sentences, gloss_list, word, lemmas=None, candidate=True, clean_flag=True, top_k=4):

        # we just add 0, we do not delete them
        if candidate:
            candidate_word = word
            if candidate_word == self.target_word or \
                    self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                    self.target_word in candidate_word:
                substituability_score = 0
                return "_"
        else:
            main_word = word
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

        if top_k is None:
            top_k = len(gloss_list)

        features = self.convert_examples_to_features(original_sentences, gloss_list, top_k, clean_flag=clean_flag)


        input_ids = torch.stack([f.input_ids for f in features]).squeeze(1)
        input_mask = torch.stack([f.input_mask for f in features]).squeeze(1)
        segment_ids = torch.stack([f.segment_ids for f in features]).squeeze(1)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)

        logits_ = F.softmax(logits[0], dim=-1)

        logits_ = logits_.detach().cpu().numpy()
        output = np.argmax(logits_, axis=0)[1]

        best_gloss = gloss_list[output]

        best_gloss = " ".join(best_gloss.split(":")[1:])
        if lemmas is not None:
            best_lemmas = lemmas[output]
            best_gloss = best_lemmas + " - " + best_gloss
        if clean_flag:
            best_gloss = self.clean(best_gloss)



        return best_gloss



    def find_best_gloss_list(self, original_sentences, gloss_list, top_k=None):
        if top_k is None:
            top_k = len(gloss_list)

        features = self.convert_examples_to_features(original_sentences, gloss_list, top_k)

        input_ids = torch.stack([f.input_ids for f in features]).squeeze(1)
        input_mask = torch.stack([f.input_mask for f in features]).squeeze(1)
        segment_ids = torch.stack([f.segment_ids for f in features]).squeeze(1)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():

                logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    labels=None)

        logits_ = F.softmax(logits[0], dim=-1)
        sort_list = torch.argsort(logits_[:, 1], dim=0, descending=True).detach().cpu().numpy()[:top_k]

        return sort_list

    def get_gloss_embedding(self, sentences):
        """
        create sentence embedding from sentence-bert
        :param sentences:  sentence
        :return: embedding
        """
        # Tokenize sentences
        sentence_embeddings = self.model_sentence.encode(sentences, convert_to_tensor=True)
        return sentence_embeddings
