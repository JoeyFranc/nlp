'''
    lbc.py
    @author Joey Franc
    
    Linguistic-Based Cues (LBC) used as statement feature representation.
    16 different features in total.
'''

import numpy as np
import nltk



THIRD_PERSON_PRONOUNS = [
    'he', 'she', 'it',
    "he'd", "she'd",
    "he'll", "she'll", "it'll",
    "he's", "she's", "it's",
    'his', 'hers', 'him', 'its',
    'they', "they're", 'them', 'their']

FIRST_PERSON_SINGULAR_PRONOUNS = ['i', "i'm", "i'd", "i'll"]
FIRST_PERSON_PLURAL_PRONOUNS = ['we', "we've", "we'll"]
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
ADJECTIVES = ['JJ', 'JJR', 'JJS']
ADVERBS = ['RB', 'RBR', 'RBS']
FUNCTION_WORDS = ['CC', 'DT', 'PDT', 'RP', 'WDT']


def _get_num_occurances(set_1, set_2):
    return sum([1 for e in set_2 if e in set_1])


class DataPoint(object):
    
    def __init__(self, datapoint):
        self.dp = datapoint
        self.words = nltk.word_tokenize(self.dp)
        self.tagged = nltk.pos_tag(self.words)
        self.pos = [tag for word, tag in self.tagged]
        self.tree = nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}').parse(self.tagged)
        
    
    def _get_num_verbs(self):
        return _get_num_occurances(VERBS, self.pos)
    
    
    def _get_num_nouns(self):
        return _get_num_occurances(NOUNS, self.pos)
    
    
#    def _get_num_clauses(self):
#        self.tree = nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}').parse(self.tagged)
#        return 
    
    
    def _get_num_sentences(self):
        return len(nltk.sent_tokenize(self.dp))
    
    
    def _get_num_words(self):
        return len(self.words)
    
    
    def _get_num_char(self):
        return len(self.dp)
    
    
    def _get_num_words_in_noun_phrases(self):
        count = 0
        for subtree in self.tree.subtrees(lambda t: t.label() == 'NP'):
            count += len(subtree.leaves())
        return count
    
    
    def _get_num_noun_phrases(self):
        return len(list(self.tree.subtrees(lambda t: t.label() == 'NP')))
    
    
    def _get_num_punctuation_marks(self):
        return _get_num_occurances(string.punctuation, self.dp)


    def _get_num_modifiers(self):
        return _get_num_occurances(ADVERBS+ADJECTIVES, self.pos)
    
    
    def _get_num_modal_verbs(self):
        return _get_num_occurances(['MD'], self.pos)
    
    
    def _get_num_uncertainty(self):
        pass
    
    
    def _get_num_other_reference(self):
        return _get_num_occurances(THIRD_PERSON_PRONOUNS, self.words)
    
    
    def _get_self_reference(self):
        return _get_num_occurances(FIRST_PERSON_SINGULAR_PRONOUNS, self.words)
    
    
    def _get_group_reference(self):
        return _get_num_occurances(FIRST_PERSON_PLURAL_PRONOUNS, self.words)
    
    
    def _get_num_adjectives(self):
        return _get_num_occurances(ADJECTIVES, self.pos)
    
    
    def _get_num_adverbs(self):
        return _get_num_occurances(ADVERBS, self.pos)
    
    
    def _get_num_unique_words(self):
        return len({word for word in self.words})
    
    
    def _get_num_function_words(self):
        return _get_num_occurances(FUNCTION_WORDS, self.pos)
    
    
#    def _get_avg_num_clauses(self):
#        return self._get_num_clauses() / self._get_num_sentences()
    
    
    def _get_avg_sentence_length(self):
        return self._get_num_words() / self._get_num_sentences()
    
    
    def _get_avg_word_length(self):
        return self._get_num_char() / self._get_num_words()
    
    
    def _get_avg_noun_phrase_length(self):
        if self._get_num_noun_phrases() == 0: return 0
        return self._get_num_words_in_noun_phrases() / self._get_num_noun_phrases()
    
    
    def _get_pausality(self):
        return self._get_num_punctuation_marks() / self._get_num_sentences()
    
    
    def _get_emotiveness(self):
        return (self._get_num_adjectives() + self._get_num_adverbs()) / (self._get_num_nouns() + self._get_num_verbs())
    
    
    def _get_lexical_diversity(self):
        return self._get_num_unique_words() / self._get_num_words()
    
    
    def _get_redundancy(self):
        return self._get_num_function_words() / self._get_num_sentences()
    
    
    def get_feature_vector(self):
        return np.array([
                self._get_num_words(),
                self._get_num_verbs(),
                self._get_num_noun_phrases(),
                self._get_num_sentences(),
#               self._get_avg_num_clauses(),
                self._get_avg_sentence_length(),
                self._get_avg_word_length(),
                self._get_avg_noun_phrase_length(),
                self._get_pausality(),
                self._get_num_modifiers(),
                self._get_num_modal_verbs(),
#               self._get_num_uncertainty(),
                self._get_num_other_reference(),
                self._get_self_reference(),
                self._get_group_reference(),
                self._get_emotiveness(),
                self._get_lexical_diversity(),
                self._get_redundancy()
        ])
    

def get_datapoints(data_set):
    return np.vstack([DataPoint(dp).get_feature_vector() for dp in data_set])



def get(train_data, test_data):
    test_set = get_datapoints(test_data['news'])
    train_set = get_datapoints(train_data['news'])
    return train_set, test_set


if __name__ == '__main__':
    get()