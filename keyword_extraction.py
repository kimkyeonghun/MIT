import ast
from collections import defaultdict, Counter
import os
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from keybert import KeyBERT
from sec_cik_mapper import StockMapper

import models.data_utils as data_utils
from log import LoggingApp

DATA_PATH = './edgar_crawler/datasets/EXTRACTED_FILINGS'
ITEM_LIST = ['item_1', 'item_7']


class KeyBERTRunner(LoggingApp):
    def __init__(self):
        super(KeyBERTRunner, self).__init__()
        self.keyModel = KeyBERT()
        self.mapper = StockMapper()
        self.ticker_list: np.ndarray = np.load(
            './ticker_data/ticker_list.npy')
        self.get_description()

    def get_description(self) -> None:
        """
            Read the concept descriptions from a file and store them in a dictionary.

            :param self: The object that this method belongs to.
            :return: None
        """
        with open("./concept_data/concept_kensho_des.txt", 'r') as f:
            data = f.readlines()

        self.description = defaultdict(str)
        for line in data:
            concept, des = line.strip().split("=")
            self.description[concept.upper()] = des
        self.concept_list = self.description.keys()

    def extract_keyword(self, contents: list) -> Counter:
        """
            Extracts keywords from the given list of contents.

            Keywords are extracted using the `keyModel.extract_keywords` method, which returns
            a list of lists of keywords, where each inner list corresponds to a content item in
            the input list. The keywords in each list are filtered to only include those with a
            score greater than or equal to the given threshold. Finally, the filtered keywords
            are counted and returned as a `Counter` object.

            Args:
                contents: A list of strings representing the contents to extract keywords from.

            Returns:
                A `Counter` object that counts the number of occurrences of each keyword in the input list of contents.
        """

        # Initialize a counter to keep track of all keywords.
        all_keywords = Counter()

        # Filter the list of contents to only include those with more than five words.
        contents = list(filter(lambda x: len(x.split()) > 5, contents))

        # Extract keywords from the filtered list of contents.
        keyword_lists = self.keyModel.extract_keywords(contents, keyphrase_ngram_range=(1, 3), stop_words='english',
                                                       use_maxsum=True, nr_candidates=15, top_n=5)

        # Iterate over the lists of keywords and update the counter of all keywords.
        for keyword_list in tqdm(keyword_lists):
            # Filter the list of keywords to only include those with a score greater than or equal to the given threshold.
            keywords = list(filter(lambda x: x[1] >= self.thres, keyword_list))
            all_keywords.update(map(lambda x: x[0], keywords))

        return all_keywords

    def get_similarity(self, keyword_embeddings: np.ndarray) -> List:
        """
            Calculates the similarity between the given keyword embeddings and a list of concept embeddings.

            Concept embeddings are obtained by calling the `keyModel.model.embed` method on a list of descriptions.
            The similarity between the keyword and concept embeddings is then calculated using the `cosine_similarity`
            function, and the resulting similarity scores are sorted and returned as a list of lists.

            Args:
                keyword_embeddings: A numpy array of keyword embeddings.

            Returns:
                A list of lists of similarity scores, where each inner list contains the similarity scores for a given concept.
        """
        # Initialize an empty list to store the similarity scores.
        new_similarity_list = []

        # Obtain the concept embeddings.
        concept_embeddings = self.keyModel.model.embed(
            list(self.description.values()))

        # Calculate the similarity between the keyword and concept embeddings.
        similarity_list = cosine_similarity(
            keyword_embeddings, concept_embeddings)

        # Transpose the similarity matrix to group the scores by concept.
        similarity_list = np.transpose(similarity_list)

        # Iterate over the rows of the transposed similarity matrix and sort the scores.
        for sims in similarity_list:
            ids = np.argsort(sims)[-5:]
            similarity = [sims[i] for i in ids][::-1]
            new_similarity_list.append(similarity)

        return new_similarity_list

    def change_concept_sec(self, sec_ticker_df: pd.DataFrame, etf_dict: dict) -> pd.DataFrame:
        """
            Transform sec_ticker_df to sec_concept_df.

            'sec_ticker_df' is a dataframe based on ticker. To concat with BERTopic results, need a dataframe based on theme(concept)
            By using etf_dict, trasnform 'sec_ticker_df' to 'sec_concept_df'

            Args:
                sec_ticker_df: A Dataframe of KeyBERT results.
                etf_dict: Dictionary about ETF-Ticker matching

            Returns:
                Same results as sec_tikcer_df from KeyBERT but differ in standards.
        """
        # Concat common and sec columns
        c2t = defaultdict(list)
        for _, row in sec_ticker_df.iterrows():
            for concept in (row['common'] + row['sec']):
                c2t[concept].append(row['ticker'])

        # By using etf_dict, re-construct dataframe
        results_df = pd.DataFrame(
            [], columns=['concept', 'etf', 'common', 'sec'])
        for concept in etf_dict.keys():
            tmp = pd.DataFrame([{
                'concept': concept,
                'etf': list((set(etf_dict[concept]) - set(c2t[concept])) & set(self.ticker_list)),
                'common': list(set(etf_dict[concept]) & set(c2t[concept])),
                'sec': list(set(c2t[concept]) - set(etf_dict[concept]))
            }])
            results_df = pd.concat([results_df, tmp], ignore_index=True)

        return results_df

    def pruning_results(self) -> Tuple[pd.DataFrame, List[float]]:
        """
        SEC keyword들과 Concept간의 유사도를 통해 filtering을 수행하고,
        이를 Kensho, ETF와 비교하여 Dataframe을 생성하는 과정
        """
        etf_dict, _ = data_utils.load_concept(True)
        rows = []

        for ticker in os.listdir('./SEC_CTM'):
            row = []
            if not ticker.endswith(f'_{self.year}_{self.thres}.txt'):
                continue

            with open(f'./SEC_CTM/{ticker}', 'r') as f:
                data = f.readlines()

            concept_list = []
            for d in data:
                # concept is upper alphabet
                concept, sims = d.strip().split(":")
                sims = np.mean(ast.literal_eval(sims))
                concept_list.append((concept, sims))

            # If concept is None because of constraints(0.5), Select concept with the largest similarity
            sorted_concept_list = sorted(
                concept_list, key=lambda x: x[1], reverse=True)
            # TODO: In now using 0.5 by default. This value is dependent on the previous step.
            concepts = list(map(lambda x: x[0], filter(
                lambda x: x[1] >= 0.5, sorted_concept_list)))

            if len(concepts) == 0:
                concepts = [sorted_concept_list[0][0]]

            row.append(ticker.split("_")[0])
            row.append(concepts)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.rename(columns={0: 'ticker', 1: 'concept'}, inplace=True)

        new_etf_dict = dict()
        for key, value in etf_dict.items():
            new_etf_dict[key.upper()] = value

        sec_ticker_df = data_utils.get_sec_df(new_etf_dict, df)
        precision = data_utils.cal_score(
            sec_ticker_df['etf'], sec_ticker_df['common'])
        recall = data_utils.cal_score(
            sec_ticker_df['sec'], sec_ticker_df['common'])
        f1 = 2*((precision*recall)/(precision+recall))
        self.logger.info(
            f"F1-score: {f1}, Precision: {precision}, Recall: {recall}")

        if not os.path.exists('sec_result'):
            os.mkdir('sec_result')

        sec_ticker_df.to_csv("{}.csv".format(
            os.path.join('sec_result', 'sec_ticker')))
        concept_sec = self.change_concept_sec(sec_ticker_df, new_etf_dict)
        concept_sec.to_csv("{}.csv".format(
            os.path.join('sec_result', 'sec_concept')))

    def process_ticker(self, ticker: str) -> None:
        """
            Process a single ticker by extracting keywords from its SEC filings and comparing them to the list of concepts.

            :param ticker: The ticker symbol to process.
            :return: None
        """
        # Extract keywords from only a subset of the content, such as the first few
        # sentences of each item.
        self.logger.info(f"{ticker} Start!")
        contents = []
        try:
            # Get Central Index Key(CIK) because SEC Filing name format is cik
            cik = str(int(self.mapper.ticker_to_cik[ticker]))
            file_lists = list(filter(lambda x: x.startswith(
                cik), os.listdir(DATA_PATH)))[-self.year:]
            for file in file_lists:
                with open(os.path.join(DATA_PATH, file)) as f:
                    data = json.load(f)
                for ITEM in ITEM_LIST:
                    # Extract keywords from the first few sentences of each item.
                    contents.extend(data[ITEM].split('\n'))
        except Exception as e:
            self.logger.info(e)
            self.logger.info(ticker)
        else:
            self.logger.info(f"Extract Keyword from {ticker} sec files")
            all_keywords = self.extract_keyword(contents)
            if len(all_keywords.keys()) == 0 or len(file_lists) == 0:
                self.logger.info(
                    f"{ticker} No 10-K report because of foreign company")
                return

            keyword_embeddings = self.keyModel.model.embed(
                list(all_keywords.keys()))

            # TODO: keyword weighting method?
            # keyword_pruned_list = list(map(lambda x: x[0], filter(lambda x: x[1]>1, all_keywords.most_common())))
            # keyword_embeddings = keyModel.model.embed(keyword_pruned_list)

            new_similarity_list = self.get_similarity(keyword_embeddings)

            self.logger.info(
                f"Get Similarity List between {ticker} and concepts")

            text = ''
            for concept, similarity in zip(self.concept_list, new_similarity_list):
                text += f"{concept}:{similarity}\n"

            if not os.path.isdir('./SEC_CTM'):
                os.mkdir('SEC_CTM')

            with open(f'./SEC_CTM/{ticker}_{self.year}_{self.thres}.txt', 'w') as f:
                f.write(text)

    def matching_concept(self, new_ticker_list=None):
        """
            Process each ticker in the ticker list and extract keywords from the corresponding SEC filings.
        """
        if new_ticker_list!=None:
            ticker_list = new_ticker_list
            self.year = 2
            self.thres = 0.25
        else:
            ticker_list = self.ticker_list
            
        for idx, ticker in enumerate(ticker_list):
            self.logger.info(
                f"{round((idx / len(ticker_list))*100, 2)}%({idx}/{len(ticker_list)})...")
            self.process_ticker(ticker)

    def start(self, **kwargs):
        os.chdir('/home/kyeonghun.kim/BERTopic')
        self.year = kwargs.get('year', 2)
        self.thres = kwargs.get('thres', 0.25)
        self.matching_concept(None)
        self.pruning_results()
