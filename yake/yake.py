# -*- coding: utf-8 -*-

"""Main module."""

import string
import os
import sys 
import jellyfish
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer 
from .Levenshtein import Levenshtein
from .datarepresentation import DataCore


nltk.data.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "nltk_data"))

class KeywordExtractor(object):

    def __init__(self, lan="en", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, filtered_n=5, features=None, stopwords=None):
        self.lan = lan

        dir_path = os.path.dirname(os.path.realpath(__file__))

        local_path = os.path.join("StopwordsList", "stopwords_%s.txt" % lan[:2].lower())

        if os.path.exists(os.path.join(dir_path,local_path)) == False:
            local_path = os.path.join("StopwordsList", "stopwords_noLang.txt")
        
        resource_path = os.path.join(dir_path,local_path)
        model_path = os.path.join(dir_path, "en_core_web_sm")

        if stopwords is None:
            try:
                with open(resource_path, encoding='utf-8') as stop_fil:
                    self.stopword_set = set( stop_fil.read().lower().split("\n") )
            except:
                print('Warning, read stopword list as ISO-8859-1')
                with open(resource_path, encoding='ISO-8859-1') as stop_fil:
                    self.stopword_set = set( stop_fil.read().lower().split("\n") )
        else:
            self.stopword_set = set(stopwords)

        self.n = n
        self.top = top
        self.filtered_n = filtered_n
        self.dedupLim = dedupLim
        self.features = features
        self.windowsSize = windowsSize
        if dedupFunc == 'jaro_winkler' or dedupFunc == 'jaro':
            self.dedu_function = self.jaro
        elif dedupFunc.lower() == 'sequencematcher' or dedupFunc.lower() == 'seqm':
            self.dedu_function = self.seqm
        else:
            self.dedu_function = self.levs

        self.entities_to_remove = ["DATE", "GPE", "CARDINAL"]
        self.NER = spacy.load(model_path)
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def jaro(self, cand1, cand2):
        return jellyfish.jaro_winkler(cand1, cand2 )

    def levs(self, cand1, cand2):
        return 1.-jellyfish.levenshtein_distance(cand1, cand2 ) / max(len(cand1),len(cand2))

    def seqm(self, cand1, cand2):
        return Levenshtein.ratio(cand1, cand2)

    def full_cleaning_pipeline(self, text):
        ner = self.NER(text)
        tokens = nltk.word_tokenize(text)
        pos_tagged_tokens = nltk.pos_tag(tokens)

        list_of_verbs = []
        for i in range(len(pos_tagged_tokens)):
            if pos_tagged_tokens[i][1].startswith("V"):
                list_of_verbs.append(pos_tagged_tokens[i][0])
        
        stopwords = nltk.corpus.stopwords.words("english")
        stopwords.extend(list_of_verbs)

        new_tokens = [token.lower() for token in tokens if token not in stopwords]
        new_text = " ".join(new_tokens)

        refined_tokens = [i.text for i in ner.ents if i.label_ in self.entities_to_remove]
        for o in refined_tokens:
            new_text = new_text.replace(o.lower(), "")
        return new_text

    def extract_keywords(self, text):
        try:
            if not(len(text) > 0):
                return []

            text = self.full_cleaning_pipeline(text.replace('\n\t',' '))
            dc = DataCore(text=text, stopword_set=self.stopword_set, windowsSize=self.windowsSize, n=self.n)
            dc.build_single_terms_features(features=self.features)
            dc.build_mult_terms_features(features=self.features)
            resultSet = []
            todedup = sorted([cc for cc in dc.candidates.values() if cc.isValid()], key=lambda c: c.H)

            if self.dedupLim >= 1.:
                return ([ (cand.H, cand.unique_kw) for cand in todedup])[:self.top]

            for cand in todedup:
                toadd = True
                for (h, candResult) in resultSet:
                    dist = self.dedu_function(cand.unique_kw, candResult.unique_kw)
                    if dist > self.dedupLim:
                        toadd = False
                        break
                if toadd:
                    resultSet.append( (cand.H, cand) )
                if len(resultSet) == self.top:
                    break

            # return [ (cand.kw,h) for (h,cand) in resultSet]
            keywords = [(cand.kw,h) for (h,cand) in resultSet]
            text_embeddings = self.model.encode(text)
            keyword_embeddings = [self.model.encode(keyword[0]) for keyword in keywords]
            cosine_scores = [cosine_similarity(i.reshape(1, -1), text_embeddings.reshape(1, -1)).item() for i in keyword_embeddings]
            top_keyword_indices = np.array(cosine_scores).argsort(axis=0)[:self.filtered_n]
            top_keywords = np.array([o[0] for o in keywords])[top_keyword_indices.tolist()].tolist()
            return top_keywords


        except Exception as e:
            print(f"Warning! Exception: {e} generated by the following text: '{text}' ")
            return []
