
'''
based on the protocol code of the BERT-based Lexical Substitution
https://aclanthology.org/P19-1328.pdf
'''
from __future__ import absolute_import, division, print_function

import logging
import re
import torch
import copy
import numpy as np
from transformers import BertTokenizer, BertModel
import string
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer



ALPHA = string.ascii_letters

logger = logging.getLogger(__name__)


class ValidationScore:
    def __init__(self, max_seq_length, do_lower_case, pre_trained="bert-large-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained(pre_trained, output_hidden_states=True, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained, do_lower_case=do_lower_case)
        self.model.to(self.device)
        self.max_seq_length = max_seq_length
        self.model.eval()
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text_a_raw, masked_id):
        pat_is = re.compile("(it|he|she|that|this|there|here) \'s", re.I)

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
        # text_a_raw = text_a_raw.replace('\'', ' ')
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
        # text_a_raw = text_a_raw.replace('\'', ' ')
        text_a_raw = text_a_raw.split(' ')
        while '' in text_a_raw:
            text_a_raw.remove('')
        text_a_raw = " ".join(text_a_raw)
        return text_a_raw

    def find_tokens(self, text_list):

        sub_tokens = 1  # clk token
        for word in text_list:
            token_list = self.tokenizer.tokenize(word)
            sub_tokens = sub_tokens + len(token_list)-1

        return sub_tokens

    def pre_processed_text(self, line, masked_id):
        masked_id = int(masked_id)
        text_a_raw = line.split(' ')  # text_a untokenized
        masked_word = text_a_raw[masked_id]  # get the target word
        text_a_raw = ' '.join(text_a_raw)

        text_a_raw, temp_index = self.clean(text_a_raw, masked_id)
        masked_word = self.clean_word(masked_word)
        masked_id = text_a_raw.index(masked_word, masked_id - temp_index)

        sub_tokens = self.find_tokens(text_a_raw[:masked_id])

        features = self.tokenizer.encode_plus(
            ' '.join(text_a_raw),
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        text_a_tokenized = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0])
        masked_word_tokenized = self.tokenizer.tokenize(masked_word)  # tokenize target word

        indexs = [i for i, word in enumerate(text_a_tokenized) if word == masked_word_tokenized[0]]
        target_word_start_index = masked_id + sub_tokens
        if self.target_start_id is not None:
            if target_word_start_index != self.target_start_id:
                print(self.target_word)
                exit(5)

        target_word_end_index = target_word_start_index + len(masked_word_tokenized) - 1

        text = ' '.join(text_a_tokenized)
        word_index = target_word_start_index

        return text, word_index, target_word_end_index, features

    def get_contextual_weights_original(self, sentences1, word, masked_id, main_word, top_k=20):
        self.attention_list = []
        self.value_list = []
        # --------------------
        self.attention_list_temp = []
        self.value_list_temp = []
        # -----------------------
        self.initial_instance = []
        self.target_word = word
        self.target_start_id = None

        sentences, target_word_start_index, target_word_end_index, features = self.pre_processed_text(sentences1,
                                                                                                      masked_id)
        masked_id = target_word_start_index
        input_ids = features['input_ids']
        input_mask = features['attention_mask']
        segment_ids = features['token_type_ids']

        self.input_mask = input_mask
        self.segment_ids = segment_ids

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            output_attention = output['attentions']

            output_attention = torch.cat(output_attention)

        # calculate the original attentions
        target_attention = output_attention[:, :, masked_id, :]
        target_attention = target_attention.view(-1, self.max_seq_length)
        target_attention = torch.mean(target_attention, axis=0)

        attention_values, attention_indices = torch.topk(target_attention, k=top_k)
        self.attention_list_temp.append(attention_indices)
        self.value_list_temp.append(attention_values)

        self.normalize_attention()  # normalize the attention-weight
        self.create_initial_instance_and_context(sentences, target_word_start_index,
                                                 target_word_end_index)  # add all the sentences where we add [MASK] token to each of the top attention token
        self.initial_embeddings(main_word)  # find embedding of each mask token list and the initial embedding of word
        return

    def normalize_attention(self, top_k=12):
        score_total = 0
        k_index = 0
        for j in range(top_k):
            if self.value_list_temp[0][j + k_index].item() <= 0.01:
                break
            score_total += self.value_list_temp[0][j + k_index].item()

        factor = 1 / score_total

        for j in range(top_k):
            if self.value_list_temp[0][j + k_index].item() <= 0.01:
                break
            self.attention_list.append(self.attention_list_temp[0][j + k_index].item())
            self.value_list.append(self.value_list_temp[0][j + k_index].item() * factor)
        return

    def create_initial_instance_and_context(self, text, target_start_id, target_end_id):

        text_a_tokenized = text.split(' ')  # text_a untokenized
        self.initial_instance.append('\t'.join([' '.join(text_a_tokenized), str(target_start_id), str(target_end_id)]))
        self.target_start_id = target_start_id
        self.target_end_id = target_end_id
        self.selected_indice = []
        selected_weight = []
        try:
            lenght_text_a = text.split(" ").index("[SEP]")
        except:
            lenght_text_a = len(text_a_tokenized)

        for i, indice in enumerate(self.attention_list):
            # if attention token in original word or after the sentence
            if int(indice) >= target_start_id and int(indice) <= target_end_id:
                continue
            if int(indice) >= lenght_text_a:
                continue
            line = copy.deepcopy(text_a_tokenized)

            if int(indice) != 0:
                line[int(indice)] = "[MASK]"

            self.initial_instance.append('\t'.join([' '.join(line)]))

            self.selected_indice.append(indice)
            selected_weight.append(float(self.value_list[i]))

        selected_weight_sum = np.sum(selected_weight)
        self.weights = [weight / selected_weight_sum for weight in selected_weight]

        text_target_masked = copy.deepcopy(text_a_tokenized)
        text_target_masked[target_start_id] = "[MASK]"
        self.initial_instance.append('\t'.join([' '.join(text_target_masked)]))
        return

    def initial_embeddings(self, main_word):
        sentences = [instance.split('\t')[0] for instance in self.initial_instance]
        target_line = self.initial_instance[0]

        input_ids = []
        attention_masks = []
        segmed_id = []

        for sentence in sentences:
            input_ids.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence.split(" "))).view(1, -1))
            attention_masks.append(self.input_mask)
            segmed_id.append(self.segment_ids)

        input_ids = torch.cat(input_ids, dim=0).type(torch.LongTensor)
        input_mask = torch.cat(attention_masks, dim=0).type(torch.LongTensor)
        segment_ids = torch.cat(segmed_id, dim=0).type(torch.LongTensor)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            embedded = output['last_hidden_state']

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

        target_start_postition = int(target_line.split('\t')[1])
        target_end_position = int(target_line.split('\t')[2])
        self.target_end_position = target_end_position
        try:
            self.target_sentence_length = target_line.split('\t')[0].split(' ').index("[SEP]")
        except:
            self.target_sentence_length = len(target_line.split('\t')[0].split(' '))
        # mean of all the positions of the words for the last hidden
        self.target_embedding_initial = torch.mean(embedded[0, target_start_postition:  target_end_position + 1, :],
                                                   axis=0)

        self.target_embedding_initial = self.target_embedding_initial.detach().cpu().numpy()
        # for each mask word (from attention score) get the embedding of each mask word for each senetence of it
        embedding_index = 1
        self.target_context_initial = embedded[embedding_index,
                                      int(self.selected_indice[0]): int(self.selected_indice[0]) + 1, :]
        embedding_index += 1

        for indice in self.selected_indice[1:]:
            self.target_context_initial = torch.cat(
                (self.target_context_initial, embedded[embedding_index, int(indice): int(indice) + 1, :]), 0)
            embedding_index += 1
        self.target_context_initial = self.target_context_initial.detach().cpu().numpy()

        return

    def get_contextual_weights_update(self, sentence, candidate_word, masked_id, main_word, top_k=20):
        # instances_candidate: embedding of each [MASK] token
        self.instances_candidate = []

        sentence, candidate_start_id, candidate_end_id, feature = self.pre_processed_text(sentence, masked_id)

        self.token_candiate_temp = feature['token_type_ids']
        self.attention_candidate_temp = feature['attention_mask']

        line_replaced_tokenized = sentence.split(" ")

        self.instances_candidate.append(
            '\t'.join([' '.join(line_replaced_tokenized), str(candidate_start_id), str(candidate_end_id)]))

        self.final_list_candidate = []
        assert (candidate_start_id == self.target_start_id)

        for indice in self.selected_indice:

            line = copy.deepcopy(line_replaced_tokenized)
            # due to the tokenization of the new word (may create multiple tokens) you may change the position
            if int(indice) > self.target_start_id:
                indice_p = int(indice) + candidate_end_id - self.target_end_id
            else:
                indice_p = int(indice)
            if indice_p > 0:
                line[indice_p] = '[MASK]'

            self.final_list_candidate.append(indice_p)
            self.instances_candidate.append('\t'.join([' '.join(line)]))

        return

    def get_val_score(self, candidate_word, rank_method='add'):
        if candidate_word == self.target_word or \
                self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                self.target_word in candidate_word:
            substituability_score = 0
            return substituability_score

        sentences = [instance.split('\t')[0] for instance in self.instances_candidate]
        candidate_line = self.instances_candidate[0]

        input_ids = []
        attention_masks = []
        segmed_id = []

        for sentence in sentences:
            input_ids.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence.split(" "))).view(1, -1))
            attention_masks.append(self.attention_candidate_temp)
            segmed_id.append(self.token_candiate_temp)

        input_ids = torch.cat(input_ids, dim=0).type(torch.LongTensor)
        input_mask = torch.cat(attention_masks, dim=0).type(torch.LongTensor)
        segment_ids = torch.cat(segmed_id, dim=0).type(torch.LongTensor)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            embedded = output['last_hidden_state']

        candidate_start_postition = int(candidate_line.split('\t')[1])
        candidate_end_position = int(candidate_line.split('\t')[2])
        # candidate_sentence_length = len(candidate_line.split('\t')[0].split(' '))

        # candidate embedding
        embedding_index = 0
        self.candidate_embedding = torch.mean(
            embedded[embedding_index, candidate_start_postition:  candidate_end_position + 1, :], axis=0)

        self.candidate_embedding = self.candidate_embedding.detach().cpu().numpy()
        embedding_index += 1

        assert (candidate_start_postition == self.target_start_id)

        candidate_context = embedded[embedding_index, self.final_list_candidate[0]: self.final_list_candidate[0] + 1, :]
        embedding_index += 1
        #
        for indice in self.final_list_candidate[1:]:
            candidate_context = torch.cat((candidate_context, embedded[embedding_index, indice: indice + 1, :]), 0)
            embedding_index += 1
        candidate_context = candidate_context.detach().cpu().numpy()

        # cosine similarity of the two embeddings
        target_candidate_similarity = self.target_sim()

        # dot product of the weights of the 12 top attention with the cosine similarity of embedding of the attetion words
        context_context_similarity = self.context_sim(candidate_context)
        context_length = candidate_context.shape[0]

        # with rank_method add the score is calculated as  (target_candidate_similarity + 2 * context_context_similarity) / 3
        substituability_score = self.score(target_candidate_similarity, context_context_similarity, context_length,
                                           rank_methode=rank_method)

        return substituability_score

    def score(self, target_candidate_similarity, context_context_similarity, context_length, rank_methode='balmult'):

        if rank_methode == 'balmult':
            return pow((target_candidate_similarity ** context_length) * context_context_similarity,
                       1 / (context_length * 2))
        elif rank_methode == 'mult':
            return pow(target_candidate_similarity * context_context_similarity, 1 / (context_length + 1))
        elif rank_methode == 'baladd':
            return (target_candidate_similarity * context_length + context_context_similarity) / (2 * context_length)
        elif rank_methode == 'add':
            return (target_candidate_similarity + 2 * context_context_similarity) / 3
        elif rank_methode == 'target_only':
            return target_candidate_similarity
        else:
            print("rank methode not valid")
            return

    def context_sim(self, candidate_context, context_methode='sim_first',
                    sim_methode='cosine_similarity', rank_methode='add'):

        target_context = self.target_context_initial
        weights = self.weights

        assert (target_context.shape == candidate_context.shape)
        context_length = target_context.shape[0]

        if context_methode == 'average_first':
            weights = np.array(weights)
            target_context = np.average(target_context, axis=0, weights=weights)
            candidate_context = np.average(candidate_context, axis=0, weights=weights)
            target_context = np.reshape(target_context, -1)
            candidate_context = np.reshape(candidate_context, -1)

            if rank_methode == 'balmult' or rank_methode == 'mult':
                return self.target_sim(target_context, candidate_context, sim_methode) ** context_length
            elif rank_methode == 'add' or rank_methode == 'baladd':
                return self.target_sim(target_context, candidate_context, sim_methode)
            else:
                print("rank methode not valid")
                return
        elif context_methode == 'sim_first':
            context_similarity = []
            for i in range(context_length):
                context_similarity.append(self.target_sim(target_context[i, :], candidate_context[i, :], sim_methode))
            if rank_methode == 'balmult' or rank_methode == 'mult':
                return np.prod(context_similarity)
            elif rank_methode == 'add' or rank_methode == 'baladd':
                weights = np.array(weights)
                context_similarity = np.array(context_similarity)
                assert (weights.shape == context_similarity.shape)
                return np.dot(context_similarity, weights)
            else:
                print("rank methode not valid")
                return
        else:
            print("context methode not valid")
            return

    def target_sim(self, target=None, candidate=None, sim_methode='cosine_similarity'):

        if target is None:
            target = self.target_embedding_initial
        if candidate is None:
            candidate = self.candidate_embedding
        if sim_methode == 'cosine_similarity':
            return self.cosine_similarity(target, candidate)
        elif sim_methode == 'dot_product':
            return np.dot(target, candidate)
        elif sim_methode == 'euclidien distance':
            return np.linalg.norm(target, candidate)
        else:
            print("sim methode not valid")
            return

    def cosine_similarity(self, vector_a, vector_b):

        nominator = np.dot(vector_a, vector_b)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

        return np.divide(nominator, denominator)
