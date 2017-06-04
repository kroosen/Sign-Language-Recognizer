import warnings
from asl_data import SinglesData
import arpa
import copy
from collections import OrderedDict



def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    word_ids = test_set.get_all_sequences().keys()
    probabilities = [dict() for w in word_ids]
    guesses = []

#    test_set contains words. Test each word with the models. this test should return a dict with the probability for each word
#   Then, the word with the highest prob is the best_guess.

    for word_id in word_ids:
        X, lengths = test_set.get_item_Xlengths(word_id)
        for candidate_word in models.keys():
            model = models[candidate_word]
            try:
                score = model.score(X, lengths)
            except:
                score = -999999
            probabilities[word_id][candidate_word] = score
    for dct in probabilities:
        guesses.append(max(dct.keys(), key=lambda x:dct[x]))

    return probabilities, guesses

def normalize_probabilities(probabilities:dict, verbose=False):
    """
    The scores of individual words can be positive or negative. To define the 
    top candidates for sentence scoring, normalize the scores between [0:1].
    :param probabilities: dict
        keys: index of word in test_set
        values: dict:
            keys: word
            values: scores for this word at this test word index
    
    :return: dict. Same as probabilities but each score is normalized [0:1]
    """
    prob_norm = []
    for i in range(len(probabilities)):
        prob_norm.append({})
        max_s = max(probabilities[i].values())
        min_s = min(probabilities[i].values())
        for w in probabilities[i].keys():
            prob_norm[i][w] = (probabilities[i][w] - min_s) / (max_s - min_s)
    
    if verbose:
        print("Scores normalized [0:1]")
        print()
#        print("Showing normalized word scores, sorted. (only top 10 for the 3 first words = 1st test sentence)...")
#        for i in range(3):
#            l = [(k,v) for k,v in zip(prob_norm[i].keys(), prob_norm[i].values())]
#            l.sort(key=lambda x: x[1], reverse=True)
#            for j in range(10):
#                print("{0:10} : {1}".format(l[j][0], l[j][1]))
#            print()
    
    return prob_norm                       

def get_candidates(probabilities: list, guesses: list, sentence: list, test_set, threshold_score=0.9, threshold_number=0, verbose=False):
    """
    For each word in the sentence, gather the candidate words with high probabilities. Each word's probability
    is divided by the top scored word's probability. Only the normalized scores that are higher than the threshold, are withheld.
    Example: if the 0gram model thinks that the word should be JOHN, then JOHN's normalized score = 1.0 and
    MARY's score could be 0.7545 (fictive number).
    
    :param probabilities: list of dicts.
        1 dict for each word in the test_set.
        Each dict contains all possible words as keys and their matching probability for the test word as values
    :param guesses: a list of str
        Best guess word from the 0gram model for each test word
    :param sentence: a list of indices of test words that form a test sentence
    :param test_set: SinglesData Object
        The built test_set (see Jupyter Notebook for more info)
    :param threshold: int or float
        The threshold for the score
    
    :return: list of dicts
        Similar to probabilities
            keys:   individual word
            values: score. The normalized score of the individual word score and the score
                    of the most probable word.
            example: {0: {'JOHN': 0.905, ...}, ...}       
    """
    # Set a threshold for the max number of candidates per word in the sentence
    # If the sentence contains more words, you want to lower the threshold for
    # computational purposes. Also, a max. cap should be set to avoid introducing
    # sentences with high log_s but too low individual word scores.
    if threshold_number == 0:
        # Set for max 1000 sentence combinations
        threshold_number = int(10000**(1./len(sentence)))
        if threshold_number > 10:
            threshold_number = 10
    
    candidates = {}
    
    for i in sentence: # for each word in the test sentence, gather the candidates
        l = []
        for k in probabilities[i].keys():
            # The normliazed word score needs to be >= threshold
            if probabilities[i][k] >= threshold_score:
                # valid candidate word for this index in the test sentence
                l.append((k, probabilities[i][k]))
        l.sort(key=lambda x:x[1], reverse=True)
        candidates[i] = {k:v for k,v in l[0:threshold_number]}
        
    if verbose:
        for i in sentence:
            print("Selected candidates for index {} (threshold = {}, threshold_number = {}):".format(i, threshold_score, threshold_number))
            print(guesses[i],":", probabilities[i][guesses[i]], "\t Correct word:", test_set.wordlist[i])
            print("============================================")
            l = [(k,v) for k,v in zip(candidates[i].keys(), candidates[i].values())]
            l.sort(key=lambda x:x[1], reverse=True)
            for tpl in l:
                print("{0:16}{1}".format(tpl[0], tpl[1]))
            print("--------------------------------------------")
            print()
    return candidates

def score_sentences(candidates, guesses, sentence, lm, verbose=False):
    """
    From the possible candidates, create all possible sentences and score them with lm.log_s
    
    :param candidates: dict of dicts
        keys: index int of the word in the test sentence
        values: dict
            keys:   individual word
            values: score. The normalized score of the individual word score and the score
                    of the most probable word.
            example: {0: {'JOHN': 0.905, ...}, ...}
    :param guesses: list of str
        Best guess words of 0gram model for the test words
    :param sentence: a list of indices of test words that form a test sentence
    :param lm: language model according to arpa module
    
    :return: dict
        keys: tuple of words that form a sentence of the test_set
        values: scores according to lm.log_s(), normalized [0:1]
    """
    def score_words_as_sentence(words:list):
        """
        Take a deepcopy of the words, remove any numbers in them and score the sentence
        according to the language model's log_s() function
        """
        words = copy.deepcopy(words)
        # remove all digits in words because they can occur in the words, but not in the LM
        for i, w in enumerate(words):
            for d in "123456789":
                if d in w[-1]:
                    # Only remove digits from the last character because in 'IX-1P', nothing should be removed!
                    words[i] = w.replace(d, "")
        # return the sentence score
        return lm.log_s(words)
              
    def build_and_score_sentences(sol, i):
        """
        recursive function to create all possible sentences from the selected words
        for each position in the sentence.
        """
        sol = copy.deepcopy(sol)
        if i < 0:
            sentence_scores[tuple(sol)] = score_words_as_sentence(sol)
            return
        for k in candidates[sentence[i]].keys():
            # Get the word of the selected candidate tuple at index i
            sol[i] = str(k)
            build_and_score_sentences(sol, i-1)
        return
    
    def normalize_sentence_scores(sentence_scores):
        """
        Normalize all sentence scores to [0:1]
        """
        min_s = min(sentence_scores.values())
        max_s = max(sentence_scores.values())
        ss_norm = {}
        for k in sentence_scores.keys():
            ss_norm[k] = (sentence_scores[k] - min_s) / (max_s - min_s)
            
#        if verbose:
#            print("SENTENCE SCORES:")
#            v = list(ss_norm.values())
#            v.sort(reverse=True)
#            for e in v:
#                print(e)

        return ss_norm
    
    if verbose:
        lens = []
        total = 1
        for k in candidates.keys():
            l = len(candidates[k].keys())
            lens.append(l)
            total *= l
        print("The number of candidates per word are: {}".format(lens))
        print("A total of {} sentences will be built and scored".format(total))
        print()
    
    sentence_scores = {}  # Empty dict to be populated by build_and_score_sentences()
    sol = [guesses[i] for i in sentence]  # Best guess sentence to start with
    build_and_score_sentences(sol, len(sentence)-1)
    
    ss_norm = normalize_sentence_scores(sentence_scores)
    
    if verbose:
        print('Scored sentences (only top 10):')
        l = [(" ".join(k), v) for k,v in zip(ss_norm.keys(), ss_norm.values())]
        l.sort(key=lambda x: x[1], reverse=True)
        for i in range(10):
            print("{0:20} ---> {1}".format(l[i][0], l[i][1]))
        print("...")
        print()
    
    return ss_norm

def combine_scores(sentence_scores, candidates, sentence, word_weight=1.0, sentence_weight=1.0, verbose=False):
    """
    For all sentences we want to score, get the sentence score and the scores
    of the individual words in the sentence.
    Then apply some magic to get a total_score and return the winner!
    :param sentence_scores: dict
        keys:   tuple of all the words in a sentence we want to test
        values: score according to the lm.log_s() function
        example: {('JOHN', 'WRITE', 'HOMEWORK'): -4.65564, ...}
    :param candidates: dict of dicts
        keys: index int of the word in the test sentence
        values: dict
            keys:   individual word
            values: score. The normliazed score of the individual word score and the score
                    of the most probable word.
            example: {0: {('JOHN', 0): 1.34, ...}, ...}
    :param sentence: a list of indices of test words that form a test sentence
    
    :return: tuple ((best_sentence, best_score), scores)
        best_sentence and best_score is the winning tuple. It is our best guess for
        a sentence from the test_set taking into account individual word scores and
        sentence scores.
        Scores is a list containing this tuple for all candidate sentences.
    """
    
    best_sentence = []
    best_score = -999999
    scores = []
    # For each sentence
    for words in sentence_scores.keys():
        s_score = sentence_scores[words]  # log_s score for sentence
        w_scores = 0.0
        # This is part of the other strategy: w_scores = []
        total_score = None
        for i in range(len(sentence)):
            # Populate the w_scores list with individual word scores
            w = words[i]
            idx = sentence[i]
            ind_normalized_score = candidates[idx][w]
            w_scores += ind_normalized_score
            # This is part of the other strategy: w_scores.append(ind_normalized_score)
            
        # HERE IS THE MEAT OF THE SCORING - this should be tuned just right
        weighted_w_scores = word_weight * w_scores
        weighted_s_score = sentence_weight * s_score
        total_score = weighted_w_scores + weighted_s_score
        # This is another strategy: total_score = sum(w_scores)/len(w_scores) * s_score
        
        if verbose:
            print("Individual word scores:",weighted_w_scores, "Sentence score:", weighted_s_score, "Total score:",total_score)
        
        scores.append((tuple(words), total_score))
        # Keep track of the winner
        if total_score > best_score:
            best_score = total_score
            best_sentence = words
#            print("New best sentence:",best_sentence)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("Showing top 10 results after combining sentence scores with individual word scores:")
        for e in scores[:10]:
            print("For words: {}, total score: {}".format(e[0], e[1]))
        print()
            
    return (best_sentence, best_score), scores
        