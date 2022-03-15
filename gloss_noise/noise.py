from src.transformers import BertTokenizer


class Gloss_noise:
    def __init__(self, do_lower_case=True, pre_trained="bert-large-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained, do_lower_case=do_lower_case)

    def test_vocab(self, golden_file):
        gold_file = open(golden_file, 'r')

        final_dict = {}
        for gold_line in gold_file:
            candidates = gold_line.split("::")[1].replace("\n", "").strip().split(";")
            for word in candidates:
                if len(word.split(" ")) > 1:
                    word_candidate = " ".join(word.split()[:-1])
                    if len(self.tokenizer.tokenize(word_candidate)) == 1:
                        pass
                    else:
                        final_dict[word_candidate] = 0
        return

    def created_proposed_list(self, change_word, wordnet_gloss, pos_tag):
        gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)
        synonyms = {}
        synonyms_final = {}

        for syn in synset:
            # adding lemmas
            for l in syn.lemmas():
                synonyms[l.name().lower()] = 0
            # adding hypernyms
            for syn_temp in syn.hypernyms():
                for l in syn_temp.lemmas():
                    synonyms[l.name().lower()] = 0
            #adding hyponyms
            for syn_temp in syn.hyponyms():
                for l in syn_temp.lemmas():
                    synonyms[l.name().lower()] = 0
        try:
            del synonyms[change_word]
        except:
            pass

        for word in synonyms:
            word_temp = word.replace("_", " ")
            word_temp = word_temp.replace("-", " ")
            word_temp = word_temp.replace("'","")
            synonyms_final[word_temp] = 0
        return synonyms_final

    def adding_noise(self, change_word, wordnet_gloss, pos_tag=None):

        gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)

        synonyms = {}
        synonyms_list = []
        for syn in synset:
            for l in syn.lemmas():
                synonyms[l.name().lower()] = 0
        try:
            del synonyms[change_word]

        except:
            pass
        for word in synonyms:
            synonyms_list.append(word)
        return synonyms_list



    def clean(self, synonyms):
        synonym_list = []
        for word in synonyms:
            token_list = self.finding_gloss.tokenizer.tokenize(word)
            if len(token_list) == 1 and token_list[0] != '[UNK]':
                synonym_list.append(token_list[0])
        return synonym_list

    def adding_synonyms(self, change_word, index_word, weak_supervision, text, wordnet_gloss, finding_gloss,
                        pos_tag=None):

        gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)
        self.finding_gloss = finding_gloss

        if weak_supervision:
            list_temp = text.split(" ")
            list_temp[index_word] = '"' + list_temp[index_word] + '"'
            text_for_gloss = " ".join(list_temp)
            for gloss in range(0, len(gloss_list)):
                gloss_list[gloss] = change_word + " : " + gloss_list[gloss]
        else:
            text_for_gloss = text

        if len(synset) == 0:
            return []
        else:
            gloss_list_sort = finding_gloss.find_best_gloss_list(text_for_gloss, gloss_list)

            synonym_flag = True
            synset_index = 0
            synonyms = []
            while (synonym_flag):
                for l in synset[gloss_list_sort[synset_index]].lemmas():
                    synonyms.append(l.name())
                try:
                    synonyms.remove(change_word)
                except:
                    pass
                if len(synonyms) > 0:
                    synonyms = self.clean(synonyms)
                if len(synonyms) > 0:
                    # synset_index = synset_index + 1
                    synonym_flag = False
                else:
                    synset_index = synset_index + 1
                if synset_index == len(gloss_list_sort):
                    break
            if synonym_flag == True:
                synonyms = []
            return synonyms
