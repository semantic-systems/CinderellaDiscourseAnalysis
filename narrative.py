import nltk
import spacy
from cleantext import clean
from typing import List
import numpy as np


class Sentence(object):
    def __init__(self, text, index, from_narrative):
        self.text = text
        self.index = index
        self.from_narrative = from_narrative

    @property
    def embedding(self):
        raise NotImplementedError

    def extract_most_similar_sentence(self, sentences: List[str]):
        raise NotImplementedError


class Narrative(object):
    def __init__(self):
        self.source = None
        self.type = None
        self.text = None
        self.uid = None

    @staticmethod
    def load_txt(path: str):
        with open(path, "r") as file:
            narrative = ''.join(file.readlines()).replace('\n', ' ')
        return narrative

    def preprocess(self, text):
        raise NotImplementedError

    def segment_documents(self, text):
        raise NotImplementedError

    @property
    def sentence_count(self):
        raise NotImplementedError

    def coreference_resolution(self, text):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
        doc = nlp(
            text,
            component_cfg={"fastcoref": {'resolve_text': True}}
        )
        return doc._.resolved_text

class CinderellaNarrative(Narrative):
    def __init__(self, path: str):
        super().__init__()
        self.source = "gutenberg"
        self.type = "fiction"
        self.raw_text = self.load_txt(path)
        self.preprocessed_text = self.preprocess(self.raw_text)
        self.segmented_text = self.segment_documents(self.preprocessed_text)

    def preprocess(self, text):
        clean_text = clean(text, fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=True, no_urls=True,
                           no_emails=True, lang="en")
        return clean_text

    def segment_documents(self, text):
        return nltk.sent_tokenize(text)

    @property
    def sentence_count(self):
        return len(self.segmented_text)

    def __str__(self):
        return self.segmented_text
