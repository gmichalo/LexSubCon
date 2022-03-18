import torch
import argparse
import numpy as np
import random
import time
from tqdm import tqdm

from reader import Reader_lexical
from metrics.evaluation import evaluation

from proposal_score.score import Cmasked
from gloss_noise.noise import Gloss_noise

from attention_score.validation_score import ValidationScore

from gloss_score.gloss_score_text import CGLOSS
from gloss_score.wordnet import Wordnet

from similarity_score_new_weight.similarity_new_predict import Similarity_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("-bep", "--bert_pretrained", type=str, help="pre-trained bert model ",
                        default="bert-base-uncased")

    # --------------- train dataset
    parser.add_argument("-tf", "--train_file", type=str, help="path of the train file dataset",
                        default='dataset/LS14/trial/coinco_trial.preprocessed')
    parser.add_argument("-gf", "--train_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS14/trial/coinco_trial.gold')

    # --------------- test dataset
    parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
                        default='dataset/LS14/test/coinco_test.preprocessed')
    parser.add_argument("-tgf", "--test_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS14/test/coinco_test.gold')

    # --------------- output results
    parser.add_argument("-outp", "--output_results", type=str, help="path of the output file with results",
                        default='dataset/results/coinco_results_gap')
    parser.add_argument("-eval", "--results_file", type=str, help="path of the output file with gap metric",
                        default='dataset/results/coinco_results_final_gap')

    # -------------------parameters
    parser.add_argument("-max_seq", "--max_seq_length", type=int, help="The maximum total input sequence length",
                        default=128)
    parser.add_argument("-max_seq_gloss", "--max_seq_length_gloss", type=int,
                        help="The maximum total input sequence length for gloss",
                        default=256)
    parser.add_argument("-lower", "--do_lower_case", type=bool, help="Whether to lower case the input text",
                        default=False)
    parser.add_argument("-ep", "--epochs", type=int, help="number of epochs for training sentence similarity bert ",
                        default=4)

    # ----hyperparameters value
    parser.add_argument("-a", "--alpha", type=float, help="alpha variable for proposed score",
                        default=1)
    parser.add_argument("-b", "--bita", type=float, help="bita variable for similarity score",
                        default=1)
    parser.add_argument("-ga", "--gamma", type=float, help="gamma variable for validation score",
                        default=1)
    parser.add_argument("-de", "--delta", type=float, help="delta variable for gloss score",
                        default=1)

    # ---------signals
    parser.add_argument("-val", "--validation_score", type=bool, help="whether we use validation score",
                        default=False)
    parser.add_argument("-pro", "--proposed_score", type=bool, help="whether we use proposed score",
                        default=True)

    parser.add_argument("-glossc", "--gloss_score", type=bool, help="whether we use gloss score",
                        default=False)
    parser.add_argument("-similn", "--similarity_sentence_new", type=bool,
                        help="whether we use similarity sentence score new",
                        default=False)
    parser.add_argument("-fna", "--fna", type=str, help="name to seperate write file",
                        default='proposed')

    # ---------------proposed score
    parser.add_argument("-glns", "--noise_type", type=str,
                        help="what noise are we using in the proposed score (GLOSS,GAUSSIAN,DROPOUT,MASKED,UNMASKED)",
                        default="GLOSS")
    # ------ gloss model
    parser.add_argument("-gc", "--gloss_checkpoint", type=str, help="path to the glossbert checkpoint",
                        default="checkpoint/gloss_bert")
    parser.add_argument("-ws", "--weak_supervision", type=bool, help="whether we use weak supervision for gloss",
                        default=True)

    # -------------------similarity flags
    parser.add_argument("-bmod", "--based_model", type=str,
                        help="name of the base model that we use in similarity (BERT, ALBERT, ROBERT)",
                        default='BERT')

    parser.add_argument("-sbep", "--similarity_pretrained_model", type=str,
                        help="pre-trained model for sentence similarities ",
                        default="bert-large-uncased")


    parser.add_argument("-tfls", "--train_flag_similarity", type=bool,
                        help="whether we want train the similarity model",
                        default=False)

    parser.add_argument("-smb", "--similarity_path_save", type=str, help="path to save/load the similarity model ",
                        default="checkpoint/similarity_new_bert/")
    # ----------gap flags
    parser.add_argument("-g", "--gap", type=bool, help="whether we use the gap ranking (candidate ranking)",
                        default=False)
    parser.add_argument("-gfc", "--golden_file_cadidates", type=str, help="path of the golden file dataset for gap",
                        default='dataset/LS14/coinco.gold.candidates')

    parser.add_argument("-seed", "--seed", type=int, help="the seed that we are using for the experiment",
                        default=6809)
    args = parser.parse_args()
    seed = args.seed

    """
    reader of features/labels and candidates if gap
    """
    reader = Reader_lexical()
    reader.create_feature(args.test_file)

    proposal_flag = True
    if args.gap:
        reader.create_candidates(args.golden_file_cadidates)
    else:
        proposal = Cmasked(args.max_seq_length, args.do_lower_case, pre_trained="bert-large-uncased")
        proposal.get_possible_words()
        proposal_flag = False
    if proposal_flag and args.proposed_score:
        proposal = Cmasked(args.max_seq_length, args.do_lower_case, pre_trained="bert-large-uncased")
        proposal.get_possible_words()

    evaluation_metric = evaluation()

    # in order to train the full model
    noise_gloss = Gloss_noise()
    wordnet_gloss = Wordnet()
    finding_gloss = None
    if (args.similarity_sentence_new or args.gloss_score):
        finding_gloss = CGLOSS(args.gloss_checkpoint, args.max_seq_length_gloss, args.do_lower_case)

    if args.validation_score:
        validation = ValidationScore(args.max_seq_length, args.do_lower_case, pre_trained="bert-large-uncased")

    alpha = args.alpha
    bita = args.bita
    gamma = args.gamma
    delta = args.delta

    "======================================================================================================"
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    "======================================================================================================"
    if args.similarity_sentence_new:
        print("initialize similiraty")
        similirity_sentence_new = Similarity_new(args.train_file, args.train_golden_file)

        if args.train_flag_similarity:
            similirity_sentence_new.train_similarity_create_dataset( noise_gloss,
                                                                    wordnet_gloss, finding_gloss)
            similirity_sentence_new.train_similarity_model(num_epochs=args.epochs,
                                                           model_save_path=args.similarity_path_save)

            similirity_sentence_new.load_model(model=args.similarity_path_save)
        else:
            similirity_sentence_new.load_model(model=args.similarity_path_save)
    "======================================================================================================"

    count_gloss = 0
    iter_index = 0
    not_found = {}
    for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:

                change_word = context[0]
                text = context[1]
                original_text = text
                index_word = context[2]
                change_word = text.split(' ')[int(index_word)]
                synonyms = []

                if args.gap:
                    try:
                        proposed_words = reader.candidates[main_word]
                    except:
                        continue
                    proposed_words = reader.created_dict_proposed(proposed_words)
                else:
                    if main_word.split('.')[0] == "":
                        word_temp = "."
                    else:
                        word_temp = main_word.split('.')[0]
                    proposed_words = noise_gloss.created_proposed_list(word_temp, wordnet_gloss,
                                                                       main_word.split('.')[-1])

                    # =============================================
                    if len(proposed_words) > 30:
                        pass

                    else:
                        if args.noise_type == "GLOSS":
                            """
                            find the probable gloss of each word
                            """
                            if len(synonyms) == 0:
                                if main_word.split('.')[0] == "":
                                    word_temp = "."
                                else:
                                    word_temp = main_word.split('.')[0]
                                synonyms = noise_gloss.adding_noise(word_temp,
                                                                    wordnet_gloss,
                                                                    main_word.split('.')[-1])
                                try:
                                    synonyms.remove(main_word.split('.')[0])
                                except:
                                    pass

                            if len(synonyms) == 0:
                                # 91- do not have wordnet synonyms in LS14
                                noise_type = "GAUSSIAN"

                        proposed_words = proposal.proposed_candidates(original_text, change_word, int(index_word),
                                                                      noise_type=args.noise_type, synonyms=synonyms,
                                                                      proposed_words_temp=proposed_words, top_k=30)
                if args.proposed_score:
                    """
                    add noise to input
                    """
                    if args.noise_type == "GLOSS":
                        """
                        find the probable gloss of each word
                        """
                        if len(synonyms) == 0:
                            if main_word.split('.')[0] == "":
                                word_temp = "."
                            else:
                                word_temp = main_word.split('.')[0]
                            synonyms = noise_gloss.adding_noise(word_temp,
                                                                wordnet_gloss,
                                                                main_word.split('.')[-1])
                        if len(synonyms) == 0:
                            # 91- do not have wordnet synonyms in LS14
                            noise_type = "GAUSSIAN"

                    """
                    add noise gloss noise mix-up noise/gaussian  noise /dropout embedding/mask/initial word
                    """


                    scores = proposal.predictions(original_text, change_word, main_word, int(index_word),
                                                  proposed_words,
                                                  noise_type=args.noise_type, synonyms=synonyms)
                    for word in proposed_words:
                        proposed_words[word] = proposed_words[word] + alpha * scores[word]

                # -------------------------------------------------------------------------------------
                if args.similarity_sentence_new:
                    """
                    get the similarity sentence similarity
                    """
                    text_similarities = []
                    similirity_sentence_new.post_tag_target(main_word)

                    for word in proposed_words:
                        similarity_score = -1
                        similarity_score = similirity_sentence_new.initial_test(word)
                        if similarity_score != 0:
                            list_temp = text.split(" ")
                            list_temp[int(index_word)] = word
                            text_update = " ".join(list_temp)
                            similarity_score = similirity_sentence_new.calculate_similarity_score(original_text,
                                                                                                  text_update)

                        proposed_words[word] = proposed_words[word] + bita * similarity_score
                # -------------------------------------------------------------------------------------
                if args.validation_score:
                    """
                    get the validation score for each word
                    """
                    validation.get_contextual_weights_original(text, change_word, index_word, main_word)
                    for word in proposed_words:
                        text_list = text.split(" ")
                        text_list[int(index_word)] = word
                        text_update = " ".join(text_list)
                        validation.get_contextual_weights_update(text_update, word, int(index_word), main_word)
                        similarity = validation.get_val_score(word)
                        proposed_words[word] = proposed_words[word] + gamma * similarity
                # -------------------------------------------------------------------------------------

                if args.gloss_score:
                    """
                    get the gloss score for each word
                    """

                    # every main word has lemma
                    finding_gloss.main_gloss_embedding = None
                    if main_word.split('.')[0] == "":
                        word_temp = "."
                    else:
                        word_temp = main_word.split('.')[0]
                    gloss_list, _, lemmas = wordnet_gloss.getSenses(word_temp, main_word.split('.')[-1])
                    # should be true
                    if args.weak_supervision:
                        list_temp = text.split(" ")
                        list_temp[int(index_word)] = '"' + list_temp[int(index_word)] + '"'
                        text_for_gloss = " ".join(list_temp)
                        for gloss in range(0, len(gloss_list)):
                            gloss_list[gloss] = change_word + " : " + gloss_list[gloss]
                    else:
                        text_for_gloss = text

                    if len(gloss_list) == 0:
                        count_gloss = count_gloss + 1
                        for word in proposed_words:
                            proposed_words[word] = proposed_words[word] + delta * 0
                    else:
                        gloss_embedding, _ = finding_gloss.find_best_gloss(text_for_gloss, gloss_list, main_word,
                                                                           lemmas,
                                                                           candidate=False, top_k=100)
                        finding_gloss.save_main_gloss_embedding(gloss_embedding)
                        for word in proposed_words:

                            gloss_list, _, lemmas = wordnet_gloss.getSenses(word)
                            if args.weak_supervision:
                                list_temp = text.split(" ")
                                list_temp[int(index_word)] = '"' + word + '"'
                                text_for_gloss = " ".join(list_temp)
                                for gloss in range(0, len(gloss_list)):
                                    gloss_list[gloss] = word + " : " + gloss_list[gloss]
                            else:
                                list_temp = text.split(" ")
                                list_temp[int(index_word)] = word
                                text_for_gloss = " ".join(list_temp)
                            if len(gloss_list) > 0:
                                _, similarity = finding_gloss.find_best_gloss(text_for_gloss, gloss_list, word, lemmas,
                                                                              candidate=True, top_k=100)
                                proposed_words[word] = proposed_words[word] + delta * similarity
                            else:
                                similarity = 0
                                proposed_words[word] = proposed_words[word] + delta * similarity

                                # -----to find words that do not have a gloss
                                if main_word.split('.')[0] == "":
                                    word_temp = "."
                                else:
                                    word_temp = main_word.split('.')[0]
                                if word_temp not in not_found:
                                    not_found[word_temp] = {}
                                not_found[word_temp][word] = {}
                if args.gap:
                    evaluation_metric.write_results(args.output_results + args.fna + "_" + str(seed) + "_gap.txt",
                                                    main_word, instance,
                                                    proposed_words)
                evaluation_metric.write_results(args.output_results + args.fna + "_" + str(seed) + "_probabilites.txt",
                                                main_word,
                                                instance,
                                                proposed_words)
                evaluation_metric.write_results_p1(args.output_results + args.fna + "_" + str(seed) + "_p1.txt",
                                                   main_word, instance,
                                                   proposed_words)
                evaluation_metric.write_results_lex_oot(args.output_results + args.fna + "_" + str(seed) + ".oot",
                                                        main_word, instance,
                                                        proposed_words, limit=10)
                evaluation_metric.write_results_lex_best(args.output_results + args.fna + "_" + str(seed) + ".best",
                                                         main_word, instance,
                                                         proposed_words, limit=1)

    end = (time.time() - start_time)
    evaluation_metric.write_time(args.output_results + args.fna + "_" + str(seed) + "_time.txt", end)
    if args.gap:
        evaluation_metric.gap_calculation(args.test_golden_file,
                                          args.output_results + args.fna + "_" + str(seed) + "_gap.txt",
                                          args.results_file + args.fna + "_" + str(seed) + "_gap.txt")
    evaluation_metric.calculation_perl(args.test_golden_file,
                                       args.output_results + args.fna + "_" + str(seed) + ".best",
                                       args.output_results + args.fna + "_" + str(seed) + ".oot",
                                       args.results_file + args.fna + "_" + str(seed) + ".best",
                                       args.results_file + args.fna + "_" + str(seed) + ".oot")
    evaluation_metric.calculation_p1(args.test_golden_file,
                                     args.output_results + args.fna + "_" + str(seed) + "_p1.txt",
                                     args.results_file + args.fna + "_" + str(seed) + "_p1.txt")
