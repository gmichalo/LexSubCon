from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
from operator import itemgetter
import unicodedata


class Reader_lexical:
    def __init__(self):
        self.words_candidate = {}
        self.final_data = {}
        self.final_data_id = {}
        self.lemmatizer = WordNetLemmatizer()

    def create_feature(self, file_train):
        # side.n	303	11	if you want to find someone who can compose the biblical side , write us .
        #
        with open(file_train, encoding='latin1') as fp:
            line = fp.readline()
            i = 0
            while line:
                context = line.split("\t")
                main_word = context[0]
                if main_word.split('.')[0] == "":
                    word = "."
                else:
                    word = main_word.split('.')[0]
                instance = context[1]
                word_index = context[2]
                sentence = self._clean_text(context[3].replace("\n", ""))
                # sentence = unicode(sentence, errors='replace')
                # sentence = context[3].replace("\n", "").replace(" , ",", ").replace(" . ",". ").replace(" ? ","? ").replace(" ; ","; ").replace(" : ",": ")
                if main_word not in self.words_candidate:
                    self.words_candidate[main_word] = {}
                if instance not in self.words_candidate[main_word]:
                    self.words_candidate[main_word][instance] = []
                self.words_candidate[main_word][instance].append([word, sentence, word_index])
                line = fp.readline()
        return

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def read_labels(self, file_gold):
        """
        bright.a 1 :: intelligent 3;clever 3;smart 1;
        bright.a 5 :: intelligent 3;clever 2;most able 1;capable 1;promising 1;sharp 1;motivated 1;
        """
        with open(file_gold,encoding="latin1") as fp:
            line = fp.readline()
            self.correct_label = {}
            self.weight_correct_label = {}
            while line:
                word = line.split("::")[0].split(" ")[0]
                word_id = line.split("::")[0].split(" ")[1]
                golden = line.split("::")[1].split(";")
                words_list = []
                weights_list = []
                pair_list = []
                for i in range(0, len(golden) - 1):
                    try:
                        word_golden, weight = golden[i].strip().split(" ")
                        words_list.append(word_golden)
                        weights_list.append(weight)
                        pair_list.append((word_golden, weight))
                    except:
                        pass
                if len(words_list) > 0:
                    if word not in self.correct_label:
                        self.correct_label[word] = {}
                    self.correct_label[word][word_id] = (words_list, weights_list)

                    if word not in self.weight_correct_label:
                        self.weight_correct_label[word] = {}
                    self.weight_correct_label[word][word_id] = pair_list

                line = fp.readline()
            return

    def read_candidates(self, candidates_file):
        target2candidates = {}
        # finally.r::eventually;ultimately
        with open(candidates_file, 'r') as f:
            for line in f:
                segments = line.split('::')
                target = segments[0]
                candidates = set(segments[1].strip().split(';'))
                target2candidates[target] = candidates
        return target2candidates

    def create_candidates(self, file_path_candidate):
        with open(file_path_candidate, encoding='latin1') as fp:
            line = fp.readline()
            main_word = ""
            self.candidates = {}
            while line:
                word = line.split("::")[0]
                # if word == "..N":

                self.candidates[word] = []
                candidates_words = line.split("::")[1]
                for candidate_word in candidates_words.split(";"):
                    if ((len(candidate_word.split(' ')) > 1) or (len(candidate_word.split('-')) > 1)) or len(candidate_word)<1:
                        pass
                    else:
                        self.candidates[word].append(candidate_word.replace("\n", ""))
                line = fp.readline()

        return

    def create_candidates_label(self, file_path_candidate):
        self.maximum_labels = 0
        self.minimum_labels = 100

        with open(file_path_candidate) as fp:
            line = fp.readline()
            main_word = ""
            self.candidates_label = {}
            while line:
                word = line.split("::")[0]
                id = word.split(" ")[1]
                word = word.split(" ")[0]
                if word not in self.candidates_label:
                    self.candidates_label[word] = {}
                self.candidates_label[word][id] = []
                candidates_words = line.split("::")[1]
                for candidate_word in candidates_words.split(";"):
                    if len(candidate_word.split(" ")) > 1:
                        candidate_word = " ".join(candidate_word.strip().split(" "))[:-1].strip()
                        # candidate_word = candidate_word.strip()

                        if ((len(candidate_word.split(' ')) > 1) or (len(candidate_word.split('-')) > 1)):
                            pass
                        else:
                            self.candidates_label[word][id].append(candidate_word.replace("\n", ""))
                if self.minimum_labels > len(self.candidates_label[word][id]):
                    self.minimum_labels = len(self.candidates_label[word][id])
                if self.maximum_labels < len(self.candidates_label[word][id]):
                    self.maximum_labels = len(self.candidates_label[word][id])
                line = fp.readline()

        return

    def clean_proposed(self, proposed):
        proposed_temp = {}
        for word in proposed:
            word_temp = word.replace("_", " ")
            word_temp = word_temp.replace("-", " ")
            if word_temp not in proposed_temp:
                proposed_temp[word_temp] = proposed[word]
        return proposed_temp

    def create_score(self, output_results, alpha, gap=True):
        eval_data = {}
        eval_file = open(output_results, 'r')
        for eval_line in eval_file:
            eval_instance_id, eval_weights = self.read_eval_line(eval_line, ignore_mwe=gap)
            eval_data[eval_instance_id] = eval_weights
        for word in eval_data:
            if word not in self.final_data:
                self.final_data[word] = {}

            # eval_data_temp = eval_data[word]
            eval_data_temp = self.clean_proposed(eval_data[word])
            for proposed in eval_data_temp:
                if proposed not in self.final_data[word]:
                    self.final_data[word][proposed] = alpha * eval_data_temp[proposed]
                else:
                    self.final_data[word][proposed] = self.final_data[word][proposed] + alpha * eval_data_temp[
                        proposed]

    def create_score_index(self, output_results):
        eval_data = {}
        eval_file = open(output_results, 'r')
        for eval_line in eval_file:
            eval_instance_id, eval_weights = self.read_eval_line(eval_line)
            eval_data[eval_instance_id] = eval_weights
        for word in eval_data:
            index = 0
            if word not in self.final_data_id:
                self.final_data_id[word] = {}
            for proposed in eval_data[word]:
                self.final_data_id[word][proposed] = index
                index = index + 1
        return

    def add_stddev(self):
        for word in self.final_data:
            scores = self.final_data[word]
            stddev = np.std(list(scores.values()))
            for candidate_word in self.final_data[word]:
                substituability_score = self.final_data[word][candidate_word]
                try:
                    candidate_id = self.final_data_id[word][candidate_word]
                except:
                    print(candidate_word)
                lm_score = (50 - candidate_id) / 50 * stddev
                substituability_score += lm_score
                self.final_data[word][candidate_word] = substituability_score
        return

    def clean_score(self):
        self.final_data = {}

    def write_final_gap(self, filepath):
        f = open(filepath, "w")

        for word in self.final_data:
            proposed_list = []

            for pword in dict(sorted(self.final_data[word].items(), key=lambda item: item[1], reverse=True)):
                proposed_list.append(pword + " " + str(self.final_data[word][pword]))
            proposed_word = "\t".join(proposed_list)
            proposed_word = proposed_word.strip()

            f.write("RESULT" + "\t" + word + "\t" + proposed_word + "\n")
        f.close()
        return

    def write_final_p1(self, filepath, limit=1):
        f = open(filepath, "w")

        for word in tqdm(self.final_data):
            proposed_list = []

            for pword in dict(sorted(self.final_data[word].items(), key=lambda item: item[1], reverse=True)[:limit]):
                proposed_list.append(pword)
            proposed_word = "\t".join(proposed_list)
            proposed_word = proposed_word.strip()

            f.write("RESULT" + "\t" + word + "\t" + proposed_word + "\n")
        f.close()
        return

    def write_final_oot(self, filepath, limit=10):
        f = open(filepath, "w")

        for word in self.final_data:
            proposed_list = []

            for pword in dict(sorted(self.final_data[word].items(), key=lambda item: item[1], reverse=True)[:limit]):
                proposed_list.append(pword)
            proposed_word = ';'.join(proposed_list)
            proposed_word = proposed_word.strip()
            f.write(word + " ::: " + proposed_word + "\n")
        f.close()

    def write_final_best(self, filepath, limit=1):
        f = open(filepath, "w")

        for word in self.final_data:
            proposed_list = []


            for pword in dict(sorted(self.final_data[word].items(), key=lambda item: item[1], reverse=True)[:limit]):
                proposed_list.append(pword)
            proposed_word = "\t".join(proposed_list)
            proposed_word = proposed_word.strip()
            f.write(word + " :: " + proposed_word + "\n")
        f.close()
        return

    def read_eval_line(self, eval_line, ignore_mwe=True):
        eval_weights = {}
        segments = eval_line.split("\t")
        instance_id = segments[1].strip()
        for candidate_weight in segments[2:]:
            if len(candidate_weight) > 0:
                delimiter_ind = candidate_weight.rfind(' ')
                candidate = candidate_weight[:delimiter_ind]
                weight = candidate_weight[delimiter_ind:]
                if ignore_mwe and ((len(candidate.split(' ')) > 1) or (len(candidate.split('-')) > 1)):
                    continue
                try:
                    eval_weights[candidate] = float(weight)
                    # eval_weights.append((candidate, float(weight)))
                except:
                    print("Error appending: %s %s" % (candidate, weight))

        return instance_id, eval_weights

    def created_dict_proposed(self, proposed_words_gap):
        proposed_words = {}

        for i in range(0, len(proposed_words_gap)):
            proposed_words[proposed_words_gap[i]] = 0

        return proposed_words

    def created_dict_proposed_train(self, main_word, proposed_words_gap, labels, initial_candidate, maximum_label=11):
        proposed_words = {}
        if main_word.split('.')[0] == "":
            word_temp = "."
        else:
            word_temp = main_word.split('.')[0]
        self.target_word = word_temp
        self.pos_initial = main_word.split(".")[-1]

        for i in range(0, len(labels)):
            candidate_word = labels[i]
            if candidate_word == self.target_word or \
                    self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                    self.target_word in candidate_word:
                pass
            else:
                proposed_words[candidate_word] = []

        j = 0
        i = len(proposed_words) - 1
        while i < maximum_label and j < len(proposed_words_gap):
            if proposed_words_gap[j] not in proposed_words:
                candidate_word = proposed_words_gap[j]
                if candidate_word == self.target_word or \
                        self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                        self.target_word in candidate_word:
                    pass
                else:
                    proposed_words[candidate_word] = []
                    i = i + 1
            j = j + 1

        j = 0
        while i < maximum_label:
            if initial_candidate[j] not in proposed_words:
                candidate_word = initial_candidate[j]
                if candidate_word == self.target_word or \
                        self.lemmatizer.lemmatize(candidate_word, self.pos_initial) == self.target_word or \
                        self.target_word in candidate_word:
                    pass
                else:
                    proposed_words[candidate_word] = []
                    i = i + 1
            j = j + 1

        return proposed_words

    def create_labels(self, labels):
        y_train = {}
        label = 1
        i = 0
        if len(labels) == 0:
            pass
        else:
            ratio = 1.0 / len(labels)
            for word in labels:
                y_train[word] = [label - i * ratio]
                i = i + 1
        return y_train



