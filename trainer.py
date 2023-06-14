import os
import copy
import warnings
from log import LoggingApp

import pandas as pd

import models.data_utils as data_utils
import models.bertopic_fun as bertopic_fun
import models.bertopic_wiki as bertopic_wiki

from transformers import pipeline

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
warnings.filterwarnings(action='ignore')


class KBERTopic(LoggingApp):
    def __init__(
        self,
        umap_n_neighbors=15,
        bertopic_min_topic_size=10,
        reduce_topic='auto',
        docu_prob=0.4,
        concept_n=10,
        threshold_sim=0.35,
        **kwargs
    ):
        super(KBERTopic, self).__init__()
        os.chdir('/home/kyeonghun.kim/BERTopic')
        self.umap_n_neighbors = umap_n_neighbors
        self.bertopic_min_topic_size = bertopic_min_topic_size
        self.reduce_topic = str(reduce_topic)
        self.docu_prob = docu_prob
        self.concept_n = concept_n
        self.threshold_sim = threshold_sim
        self.load_mode = kwargs.get('load_mode', False)
        self.model_name = kwargs.get('model_name', '')
        self.kensho = kwargs.get('kensho', False)
        self.wiki = kwargs.get('wiki', False)

        self.logger.info("Start Loading files to use BERTopic")
        self.concept_dict, self.concept_list = data_utils.load_concept(
            self.kensho)
        self.df = data_utils.load_news_dataset(self.kensho, only_news=True)
        self.tickers = self.df.ticker.apply(lambda x: x.split(',')).tolist()

        total_tic = []
        for row in self.tickers:
            total_tic.extend(row)
        self.total_ticker = set(total_tic)

        tic2com = data_utils.load_tic2com()
        self.df = data_utils.data_preprocessing(self.df, tic2com)
        self.docs = self.df['content'].tolist()

        self.logger.info("Load zero-shot learning")
        self.classifier = pipeline('zero-shot-classification',
                                   model='facebook/bart-large-mnli', device=0)

    def load_model(self):
        if self.load_mode:
            if len(self.model_name):
                topic_model = bertopic_fun.load_BERTopic_model(self.model_name)
                topic, probs = topic_model.transform(self.docs)
            else:
                raise ValueError("Need model name when you set load_mode")
        else:
            if self.reduce_topic.isdigit:
                self.reduce_topic = int(self.reduce_topic)
            topic_model, topic, probs, model_path = bertopic_fun.train_and_save_model(
                self.docs, self.umap_n_neighbors, self.bertopic_min_topic_size, self.reduce_topic)
            self.logger.info(f"BERTopic Model Path: {model_path}")

        return topic_model, topic, probs

    def run(self):
        self.logger.info("Load pre-trained model or train new model")
        topic_model, topic, probs = self.load_model()

        self.logger.info("Matching Ticker to Topic")
        BERTopic_concept, BERTopic_ticker = bertopic_fun.filtering_tic_and_com(
            topic, probs, self.df, self.total_ticker, self.docu_prob)

        self.logger.info("Matching Concept to Ticker")

        # TODO:합쳐도 될 것 같은데..?
        if self.kensho:
            concept_path, concept2news, scores = bertopic_fun.kensho_concept2ticker(
                topic_model, self.concept_dict, BERTopic_ticker, self.concept_n, self.threshold_sim, self.classifier)
        else:
            results_df, concept2news, concept_path, scores = bertopic_fun.concept2ticker(
                topic_model, self.concept_dict, self.total_ticker, BERTopic_ticker, self.concept_n, self.threshold_sim, self.classifier)

        self.logger.info(f"Saveing path : {concept_path}")

        self.logger.info(
            f"F1-score: {scores[0]}, Precision: {scores[1]}, Recall: {scores[2]}")

        # company_id 알아야 함
        news_final_df = data_utils.matching_news_concept(
            self.df, topic, concept2news)
        self.logger.info("Finish matching ticker to concept in News")

        news_final_df.to_csv(os.path.join(concept_path, 'develop_final.csv'))
