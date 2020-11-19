from nltk.corpus import wordnet as wn

similarity_heuristic_dict = ["path_similarity", "lch_similarity", "wup_similarity",
                             "res_similarity", "jcn_similarity", "lin_similarity"]


def get_all_word_senses(context, context_pos):
    context_word_senses = []
    for (c, pos) in zip(context, context_pos):
        context_word_senses.append(wn.synsets(c, pos))
    return context_word_senses


def extract_correct_word_senses(context_pos, context_word_senses, target_word_index):
    # Keep only word senses that have the same POS
    result = []
    for (i, pos) in enumerate(context_pos):
        if (i != target_word_index):
            result.append(context_word_senses[i])

    return result


def calculate_best_score(target_word_sense, context_word_senses, similarity_name):
    best_lch_score = 0
    for context_word_sense in context_word_senses:
        score = 0
        for word_sense in context_word_sense:

            similarity_score = calculate_similarity_score(
                target_word_sense, word_sense, similarity_name)

            score = max(similarity_score, score)
        best_lch_score += score
    return best_lch_score


def calculate_similarity_score(target_word_sense, word_sense, similarity_name):

    similarity_score = 0

    try:
        if (similarity_name == "path_similarity"):
            similarity_score = target_word_sense.path_similarity(word_sense)
        elif (similarity_name == "lch_similarity"):
            similarity_score = target_word_sense.lch_similarity(word_sense)
        elif (similarity_name == "wup_similarity"):
            similarity_score = target_word_sense.wup_similarity(word_sense)
        elif (similarity_name == "res_similarity"):
            similarity_score = target_word_sense.res_similarity(word_sense)
        elif (similarity_name == "jcn_similarity"):
            similarity_score = target_word_sense.jcn_similarity(word_sense)
        elif (similarity_name == "lin_similarity"):
            similarity_score = target_word_sense.lin_similarity(word_sense)
    except:
        return 0

    if (similarity_score is None):
        similarity_score = 0

    return similarity_score


def get_best_word_sense(context, context_pos, context_word_senses, target_word, similarity_name):

    target_word_index = context.index(target_word)

    # Get target word senses
    target_word_senses = context_word_senses[target_word_index]

    if (len(target_word_senses)) == 0:
        return None

    # Remove target word
    context_word_senses = extract_correct_word_senses(
        context_pos, context_word_senses, target_word_index)

    # Compare by similarity
    target_word_senses_scores = []
    for target_word_sense in target_word_senses:
        target_word_sense_score = calculate_best_score(
            target_word_sense, context_word_senses, similarity_name)
        target_word_senses_scores.append(target_word_sense_score)

    # Max score Index
    target_word_sense_max_score = max(target_word_senses_scores)

    target_best_word_sense = target_word_senses[target_word_senses_scores.index(
        target_word_sense_max_score)]

    return target_best_word_sense


if __name__ == '__main__':
    context = ["U.N.", "group", "draft", "plan", "reduce", "emission"]
    context_pos = ["n", "n", "v", "n", "v", "n"]
    context_word_senses = get_all_word_senses(context, context_pos)

    similarity_name = "lcn_similarity"

    for (target_word, target_word_pos) in zip(context, context_pos):
        print(f"{target_word} - {target_word_pos}")
        result = get_best_word_sense(
            context, context_pos, context_word_senses, target_word, similarity_name)

        print(result)
