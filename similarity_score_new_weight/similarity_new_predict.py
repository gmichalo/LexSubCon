# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function


import torch
from torch.utils.data import DataLoader
import math

import logging
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer

from similarity_score_new_weight.translation import back_translation
from reader import Reader_lexical

import sys

sys.path.insert(0, 'similarity_score_new_weight/sentence/sentence-transformers/')
from sentence_transformers_new import SentenceTransformer, LoggingHandler, losses, util, InputExample

# from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import re
import nltk
from nltk.corpus import wordnet

logger = logging.getLogger(__name__)


class Similarity_new:
    def __init__(self, train_file, golden_file, model="stsb-roberta-large"):
        self.reader = Reader_lexical()
        self.reader.create_feature(train_file)
        self.reader.read_labels(golden_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        self.lemmatizer = WordNetLemmatizer()

        self.translation_module = back_translation()

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
        # new_text = pat_s.sub("", new_text)
        # new_text = pat_s2.sub("", new_text)
        new_text = pat_not.sub(" not", new_text)
        new_text = pat_would.sub(" would", new_text)
        new_text = pat_will.sub(" will", new_text)
        new_text = pat_am.sub(" am", new_text)
        new_text = pat_are.sub(" are", new_text)
        new_text = pat_ve.sub(" have", new_text)
        # text_a_raw = new_text.replace('\'', ' ')
        text_a_raw = text_a_raw.split(' ')
        while '' in text_a_raw:
            text_a_raw.remove('')
        text_a_raw = " ".join(text_a_raw)

        return text_a_raw

    def calculate_similarity_score(self, original_sentences, update_sentence):

        original_embedding = self.get_sentence_embedding(self.clean(original_sentences).lower())
        update_embedding = self.get_sentence_embedding(self.clean(update_sentence).lower())
        # what is used in the sentencebert Semantic Textual Similarity
        cosine = util.pytorch_cos_sim(original_embedding, update_embedding)[0].item()

        return cosine

    def get_sentence_embedding(self, sentences):
        """
        create sentence embedding from sentence-bert
        :param sentences:  sentence
        :param tokenizer: pre-trained tokenizer
        :param model: pre-trained model
        :return: embedding
        """
        # Load AutoModel from huggingface model repository using best model for Semantic Textual Similarity

        # Tokenize sentences
        sentence_embeddings = self.model_sentence.encode(sentences,  convert_to_tensor=True)
        return sentence_embeddings

    def train_similarity_create_dataset(self,  noise_gloss, wordnet_gloss,
                                        finding_gloss):
        bert_dataset = []

        train_dataset = self.reader.words_candidate
        labels = self.reader.correct_label
        weight_pair = self.reader.weight_correct_label

        count1 = 0
        count2 = 0

        for main_word in tqdm(train_dataset):
            for instance in train_dataset[main_word]:
                for context in train_dataset[main_word][instance]:
                    text_temp = context[1]
                    try:
                        word_labels = labels[main_word][instance][0]
                        weight_labels = labels[main_word][instance][1]
                    except:
                        count2 = count2 + 1
                        continue
                    context[1] = context[1] + " "
                    text = context[1]
                    original_text = text
                    translated_sentence = self.translation_module.get_translate_example(self.clean(original_text))

                    # get the pair of original sentence with the update sentence with the candidate word per weight
                    final = 1
                    for word_index in range(0, len(word_labels)):
                        label = word_labels[word_index]
                        temp_text = text_temp.split(" ")
                        temp_text[int(context[2])] = label
                        pair_text = " ".join(temp_text)

                        bert_dataset.append(
                            [self.clean(original_text).lower(), self.clean(pair_text).lower(), final])

                    synonyms = noise_gloss.adding_synonyms(context[0], int(context[2]), True,
                                                           original_text,
                                                           wordnet_gloss, finding_gloss, main_word.split('.')[-1])



                    for synonym in synonyms:
                        if synonym not in word_labels:
                            temp_text = text_temp.split(" ")
                            temp_text[int(context[2])] = synonym
                            pair_text = " ".join(temp_text)
                            bert_dataset.append(
                                [self.clean(original_text).lower(), self.clean(pair_text).lower(), 0.9])

                    # not add the pair original and translate sentence
                    indices = [index for index, element in enumerate(translated_sentence.split(" ")) if
                               element == context[0]]
                    if len(indices) > 0:
                        min_value = -1
                        min_distance = 100
                        for index in indices:
                            if abs(index - int(context[2])) < min_distance:
                                min_value = index
                                min_distance = abs(index - int(context[2]))

                        final = 1
                        if min_value != -1:
                            for label in word_labels:
                                temp_text = translated_sentence.split(" ")
                                temp_text[min_value] = label
                                pair_text = " ".join(temp_text)
                                bert_dataset.append(
                                    [self.clean(pair_text).lower(), self.clean(translated_sentence).lower(), 1.0])

                            for synonym in synonyms:
                                if synonym not in word_labels:
                                    temp_text = translated_sentence.split(" ")
                                    temp_text[min_value] = synonym
                                    pair_text = " ".join(temp_text)
                                    bert_dataset.append(
                                        [self.clean(pair_text).lower(), self.clean(translated_sentence).lower(),
                                         0.9])
                        count1 = count1 + 1



        self.x_train = bert_dataset
        return

    def train_similarity_model(self, model_name="stsb-roberta-large", batch_size=4,
                               num_epochs=4, model_save_path="checkpoint/similarity_new_bert/"):

        # Read the dataset
        model_name = model_name
        train_batch_size = batch_size
        num_epochs = num_epochs
        model_save_path = model_save_path

        model = SentenceTransformer(model_name)

        train_samples = []
        dev_samples = []
        # dev_flag = int(0.8 * len(self.x_train))

        i = 0
        for pair in self.x_train:
            inp_example = InputExample(texts=[pair[0], pair[1]], label=float(pair[2]))
            train_samples.append(inp_example)

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model=model)

        # Development set: Measure correlation between cosine score and gold labels
        # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        # warmup_steps = 0

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path)

    def pos_tag(self, word):
        candidate_word = word
        substituability_score = -1
        if candidate_word == self.target_word or \
                self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                self.target_word in candidate_word:
            substituability_score = 0
        return substituability_score

    def initial_test(self, word):
        candidate_word = word
        substituability_score = -1
        if candidate_word == self.target_word or \
                self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                self.target_word in candidate_word:
            substituability_score = 0
        return substituability_score

    def post_tag_target(self, word):
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
        return

    def load_model(self, model="stsb-roberta-large"):
        self.model_sentence = SentenceTransformer(model)
        self.model_sentence.to(self.device)
        self.model_sentence.eval()