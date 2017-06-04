import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError()

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = 999999
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type='diag', \
                                        random_state=self.random_state, n_iter=1000, \
                                        verbose=False).fit(self.X, self.lengths)
                # BIC score is defined as -2*log(L) + p*log(N) where
                #   L: likelihood
                #   p: number of parameters
                #       n: number of states
                #       d: number of features
                #   N: number of data points (examples)
                logL = hmm_model.score(self.X, self.lengths)
                logN = np.log(len(self.sequences))
                p = n**2 + 2*len(self.sequences[0])*n - 1
                score = -2*logL + p*logN
                if score < best_score:
                    best_score = score
                    best_model = hmm_model
            except:
                pass
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_model = None
        best_score = -99999999
        words = self.words.keys()
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type='diag', \
                                        random_state=self.random_state, n_iter=1000, \
                                        verbose=False).fit(self.X, self.lengths)
                score_other_words = []
                for word in words:
                    if word == self.this_word:
                        score_this_word = hmm_model.score(self.X, self.lengths)
                    else:
                        X, lengths = self.hwords[word]
                        try:
                            score_other_words.append(hmm_model.score(X, lengths))
                        except:
                            pass
                # DIC = P(X|hmm_model) - average(P(Y|hmm_model) for all words Y except X)
                # In words: the chance of getting the right word - the  chance of getting the wrong word
                DIC = score_this_word - sum(score_other_words) / len(score_other_words)
                if DIC > best_score:
                    best_score = DIC
                    best_model = hmm_model
            except:
                pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = self.min_n_components
        best_model = None
        best_score = -999999
        for n in range(self.min_n_components, self.max_n_components + 1):    
            score = []
            if len(self.sequences) >= 3: n_splits = 3
            elif len(self.sequences) == 2: n_splits = 2
            else:
#                print("For word {}: only {} sample available --> fail".format(self.this_word, len(self.sequences)))
                next
            try:
                split_method = KFold(n_splits=n_splits)
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        X, lengths = combine_sequences(cv_train_idx, self.sequences)    
                        hmm_model = GaussianHMM(n_components=n, covariance_type='diag', \
                                                n_iter=1000, random_state=self.random_state, \
                                                verbose=False).fit(X, lengths)
                        X, lengths = combine_sequences(cv_test_idx, self.sequences)
                        score.append(hmm_model.score(X, lengths))
                    except:
                        pass
                if len(score) > 0:
                    avg_score = sum(score)/float(len(score))
                    if avg_score > best_score:
                        best_score = avg_score
                        best_num_components = n
                        best_model = hmm_model
            except:
                pass
        if self.verbose:
            print("model created for {} with {} states".format(self.this_word, best_num_components))
        return best_model