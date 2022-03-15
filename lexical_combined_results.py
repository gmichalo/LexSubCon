from reader import Reader_lexical
import argparse
from metrics.evaluation import evaluation

"""
code for the combination of the features if we run them individually
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gap", type=bool, help="whether we use the gap ranking (candidate ranking)",
                        default=False)
    parser.add_argument("-gfc", "--golden_file_cadidates", type=str, help="path of the golden file dataset for gap",
                        default='dataset/LS07/lst.gold.candidates')
    # --------------- test dataset
    parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
                        default='dataset/LS07/test/lst_test.preprocessed')
    parser.add_argument("-tgf", "--test_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS07/test/lst_test.gold')


    # parser.add_argument("-gfc", "--golden_file_cadidates", type=str, help="path of the golden file dataset for gap",
    #                     default='dataset/LS14/coinco.gold.candidates')
    # # --------------- test dataset
    # parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
    #                     default='dataset/LS14/test/coinco_test.preprocessed')
    # parser.add_argument("-tgf", "--test_golden_file", type=str, help="path of the golden file dataset",
    #                     default='dataset/LS14/test/coinco_test.gold')
    # #


    # --------------- output results
    parser.add_argument("-outp", "--output_results", type=str, help="path of the output file with results",
                        default='dataset/results/results_gap')
    parser.add_argument("-eval", "--results_file", type=str, help="path of the output file with gap metric",
                        default='dataset/results/results_final_gap')

    args = parser.parse_args()

    reader = Reader_lexical()
    evaluation_metric = evaluation()

    reader.create_feature(args.test_file)
    reader.read_labels(args.test_golden_file)
    if args.gap:
        reader.create_candidates(args.golden_file_cadidates)

    #-------------------------------------------ls07-----------------------------------------------------
    reader.create_score("dataset/results/test_final/proposal/results_gapproposed_6809_probabilites.txt", 0.05, args.gap)
    reader.create_score("dataset/results/test_final/gloss/results_gapproposed_6809_probabilites.txt",0.05, args.gap)
    reader.create_score("dataset/results/test_final/similarity/results_gapproposed_6809_probabilites.txt",1.0, args.gap)
    reader.create_score("dataset/results/test_final/validation/results_gapproposed_6809_probabilites.txt", 0.5, args.gap)


    reader.write_final_gap(args.output_results + "_gap.txt")
    reader.write_final_p1(args.output_results + "_p1.txt")
    reader.write_final_best(args.output_results + ".best")
    reader.write_final_oot(args.output_results + ".oot")



    if args.gap:
        evaluation_metric.gap_calculation(args.test_golden_file, args.output_results + "_gap.txt",
                                          args.results_file + "_gap.txt")
    evaluation_metric.calculation_perl(args.test_golden_file, args.output_results + ".best",
                                       args.output_results + ".oot", args.results_file + ".best",
                                       args.results_file + ".oot")
    evaluation_metric.calculation_p1(args.test_golden_file, args.output_results + "_p1.txt",
                                     args.results_file + "_p1.txt")
