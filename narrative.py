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


if __name__ == "__main__":
    import re
    import requests
    # from gutenberg.cleanup import strip_headers
    from gutenberg.query import get_metadata
    # from gutenberg.acquire import get_metadata_cache
    import gutenbergpy
    from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheTypes


    def get_gutenberg_cache():
        cache = get_metadata_cache()
        cache.populate()

    def get_gutenbergpy_cache():
        GutenbergCache.create(type=GutenbergCacheTypes.CACHE_TYPE_MONGODB)

    def download_cinderella_stories():
        get_gutenbergpy_cache()
        # Get all EText IDs
        cache = GutenbergCache.get_cache()
        etext_ids = cache.query(titles=['Cinderella', "cinderella"])
        print(etext_ids)
        print(f"{len(etext_ids)} Cinderellas found.")
        column_query = "PRAGMA table_info(books)"
        columns = cache.native_query(column_query)
        column_names = [column[1] for column in columns]
        print(column_names)
        print(cache.query(downloadtype=['application/plain', 'text/plain', 'text/html; charset=utf-8'],
                          titles="cinderella"))

        # Search for Cinderella-related books
        cinderella_books = cache.native_query({"title":'Cinderella'})

        print("cinderella_books", cinderella_books)

        # Filter out non-Cinderella stories
        cinderella_variants = []
        for book in cinderella_books:
            if 'Cinderella' in book.title:
                cinderella_variants.append(book)

        # Download the Cinderella stories
        for book in cinderella_variants:
            gutenbergpy.download(book.gutenberg_id, overwrite=False)

        for etext_id in etext_ids:
            # Fetch the metadata for the EText
            metadata = get_metadata('title', etext_id)
            title = metadata[0]['title']

            # Fetch the text content of the EText
            text = strip_headers(load_etext(etext_id)).strip()
            subname = re.sub(r"\W+", "_", title)

            # Check if the text contains "Cinderella" in the subject field
            if 'Cinderella' in metadata[0].get('subject', ''):
                # Save the Cinderella story as a text file
                filename = f'{etext_id}_{subname}.txt'
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(text)

    # Run the function to download Cinderella stories
    # get_gutenbergpy_cache()
    download_cinderella_stories()

    #     # Get all EText IDs
    #     etext_ids = get_etexts('title', value='Cinderella')
    #
    #     for etext_id in etext_ids:
    #         # Fetch the metadata for the EText
    #         metadata = get_metadata('title', etext_id)
    #         title = metadata[0]['title']
    #
    #         # Check if the title contains "Cinderella"
    #         if re.search(r'\bCinderella\b', title, re.IGNORECASE):
    #             # Fetch the text content of the EText
    #             text = strip_headers(load_etext(etext_id)).strip()
    #             subname = re.sub(r"\W+", "_", title)
    #             # Save the Cinderella story as a text file
    #             filename = f'{etext_id}_{subname}.txt'
    #             with open(f"./narratives/cinderella/{filename}", 'w', encoding='utf-8') as file:
    #                 file.write(text)
    #
    #
    # # Run the function to download Cinderella stories
    # download_cinderella_stories()