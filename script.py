from util import evaluate, recall

""" Table 7.1: Evaluation metrics on the English-French dataset at feature level """
evaluate(filename="./back_translation/bt_en_fr_features.csv", method="back_translation")
print("")
evaluate(filename="./dependency_tree/dt_en_fr_features.csv", method="dependency_tree")
print("")
evaluate(filename="./word_distribution/wd_en_fr_features.csv", method="word_distribution")
print("")
evaluate(filename="./word_embedding/we_en_fr_features.csv", method="word_embedding")

""" Table 7.2: Evaluation metrics on the German dataset at feature level """
recall("./results/dependency_tree/dt_german_feature_level.txt")
print("")
recall("./results/word_distribution/wd_german_feature_level.txt")
print("")
recall("./results/word_embedding/we_german_feature_level.txt")

""" Table 7.3: Evaluation metrics on the Japanese dataset at feature level """
recall("./results/dependency_tree/dt_japanese_feature_level.txt")
print("")
recall("./results/word_distribution/wd_japanese_feature_level.txt")
print("")
recall("./results/word_embedding/we_japanese_feature_level.txt")


""" Table 7.4: Evaluation metrics on the German dataset at chapter level """
recall("./results/dependency_tree/dt_german_chapter_level.txt")
print("")
recall("./results/word_distribution/wd_german_chapter_level.txt")
print("")
recall("./results/word_embedding/we_german_chapter_level.txt")


""" Table 7.5: Evaluation metrics on the Japanese dataset at chapter level """
recall("./results/dependency_tree/dt_japanese_chapter_level.txt")
print("")
recall("./results/word_distribution/wd_japanese_chapter_level.txt")
print("")
recall("./results/word_embedding/we_japanese_chapter_level.txt")
