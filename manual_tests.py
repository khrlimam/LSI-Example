import math


class TFIDF:
    def __init__(self, *args):
        self.docs = args
        self.tokenized_doc = [d.lower().split() for d in self.docs]
        self.enumerated_doc = enumerate(self.docs)
        self.terms = set().union(*self.tokenized_doc)
        self.term_counters()
        self.collection_length = self.docs.__len__()

    def term_counters(self):
        term_counters = tuple()
        for index, doc in self.enumerated_doc:
            terms_ = set(doc.lower().split())
            counters = {term: doc.lower().split().count(term) for term in terms_}
            term_counters += ((index, counters),)
        self.term_counters_ = term_counters

    def idf_counters(self):
        idf_counters = dict()
        for term in self.terms:
            idf_counters[term] = sum([self.get_term_counts(item[1], term) for item in self.term_counters_])
        return idf_counters

    def tfidf(self):
        tfidf = list()
        for index, tokenized_doc in enumerate(self.tokenized_doc):
            set_td = set(tokenized_doc)
            tfidf_current_doc = list()
            for term in set_td:
                count_term_appearance_in_doc = filter(lambda item: item > 0,
                                                      [self.get_term_counts(item[1], term) for item in
                                                       self.term_counters_])
                tf = float(self.get_term_counts(self.term_counters_[index][1], term)) / float(
                    self.tokenized_doc[index].__len__())
                idf = math.log10(float(self.collection_length) / float(count_term_appearance_in_doc.__len__()))
                tfidf_value = tf * idf
                tfidf_current_doc.append(tfidf_value)
            tfidf.append(tfidf_current_doc)
        return tfidf

    def get_term_counts(self, term_counter, term):
        count = term_counter.get(term)
        if count is None:
            return 0
        return count


def sim(q, d):
    return ((q[0] * d[0]) + (q[1] * d[1])) / (math.sqrt(math.pow(q[0], 2) + math.pow(q[1], 2)) * math.sqrt(
        math.pow(d[0], 2) + math.pow(d[1], 2)))
