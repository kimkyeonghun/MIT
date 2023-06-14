from collections import defaultdict, Counter
import copy
import datetime
import enum
import math
import os
from typing import Tuple, List
from tqdm import tqdm

from bertopic import BERTopic
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from utils import make_date_dir
from models.data_utils import save_file, cal_score


def load_BERTopic_model(model_name: str) -> BERTopic:
    """
        Load a previously trained BERTopic model from a saved file.

        Args:
        model_name (str): The name of the saved model file to be loaded.

        Returns:
        BERTopic: The loaded BERTopic model.
    """
    topic_model: BERTopic = BERTopic.load(os.path.join('./model_save',model_name))
    try:
        topic_model.get_topic_info()
    except:
        topic_model.custom_labels = topic_model.generate_topic_labels()
    
    return topic_model


def train_and_save_model(docs: list, umap_n_neighbors: int, bertopic_min_topic_size: int, reduce_topic: enum) -> Tuple[BERTopic, List[int], np.ndarray, str]:
    """
        Train a BERTopic model using the specified hyperparameters, and save the trained model to disk.
        BERTopic can be tuned using UMAP, HDBSCAN, and CountVectorizer.
        More information on these hyperparameters can be found in the official BERTopic documentation.

        If the reduce_topic parameter is set to 'auto', the number of topics will be automatically reduced to an optimal value.
        This is done by training the model using 'auto' mode, and then recursively training the model with the optimal number
        of topics until the number of topics is between 100 and 200.

        More information can be found in the official BERTopic document.

        When reduce topic number using 'auto' mode, -1(outlier) topic increase a lot.
        Therefore, you need to find optimal topic number using recursive 'auto' mode and train again usin optimal topic number
        
        Args:
            docs (list): A list of documents to train the model on.
            umap_n_neighbors (int): The number of neighbors to use when training the UMAP model.
            bertopic_min_topic_size (int): The minimum size of a topic when training the BERTopic model.
            reduce_topic (enum): The method to use for reducing the number of topics. Can be an integer or the string 'auto'.

        Returns:
            Tuple[BERTopic, List[int], np.ndarray, str]:
            A tuple containing the trained BERTopic model, the list of topics, the probabilities of each topic,
            and the path to where the model was saved.
    """
    umap_model = UMAP(n_neighbors=umap_n_neighbors,
                      n_components=5, min_dist=0.0, metric='cosine')
    vectorizer_model = CountVectorizer(
        stop_words="english", ngram_range=(1, 3), min_df=20)

    topic_model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer_model,
                           min_topic_size=bertopic_min_topic_size, verbose=True, diversity=0.8)
    if isinstance(reduce_topic, str):
        while not (100 <= len(np.unqiue(np.array(topics))) < 200):
            topics, _ = topic_model.fit_transform(docs)
            topics, _ = topic_model.reduce_topics(
                docs, topics, nr_topics='auto')
        optim_topic = len(np.unqiue(np.array(topics)))
        topic_model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer_model,
                               min_topic_size=bertopic_min_topic_size, verbose=False, diversity=0.8, nr_topics=optim_topic)
        topics, probs = topic_model.fit_transform(docs)

    else:
        topic_model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer_model,
                            min_topic_size=bertopic_min_topic_size, verbose=True, diversity=0.8, nr_topics=reduce_topic)
        topics, probs = topic_model.fit_transform(docs)
    model_path = './model_save'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    name = 'topic_model'
    today = datetime.datetime.now()
    name += "_"+today.strftime('%Y%m%d')
    model_path = os.path.join(model_path, name)
    topic_model.save(model_path)
    
    return topic_model, topics, probs, model_path


def filtering_tic_and_com(topics: list, probs: list, df: pd.DataFrame, total_tic: set, docu_prob: float, mean_prob=None) -> Tuple[dict, dict]:
    """
        Using fitted BERTopic model, transform dataset and get (topic, prob) pairs.

        Filtering: if one docu's prob is lower than docu_prob, it can be interpreted uncertain topic.
        Gathering: After counting by topic, if number of docus is small, it might be outlier topic.
        Matching: match BERTopic_concept(ticker-topics) and BERTopic_ticker(topic-tickers)

        Args:
          topics: list of topics
          probs: list of probabilities
          df: Pandas DataFrame containing the data
          total_tic: set of total tickers
          docu_prob: float indicating the document probability
          mean_prob: float indicating the mean probability (defaults to None)

        Returns:
           Tuple containing two dictionaries: BERTopic_concept and BERTopic_ticker
        """
    if mean_prob == None:
        mean_prob = docu_prob

    BERTopic_ticker, BERTopic_concept = defaultdict(list), defaultdict(list)
    for ticker in list(total_tic):
        docs = df[df['ticker'].apply(lambda x: ticker in x.split(','))].content.tolist()
        sub_topics = [topics[i] for i in df[df['ticker'].apply(lambda x: ticker in x.split(','))].index]
        sub_probs = [probs[i] for i in df[df['ticker'].apply(lambda x: ticker in x.split(','))].index]
        topic_prob = list(zip(sub_topics, sub_probs))
        # docu가 topic에 할당될 확률이 낮은 경우를 filtering
        topic = np.array(sub_topics)[list(
            filter(lambda i: sub_probs[i] >= docu_prob, range(len(sub_probs))))]

        c = Counter(topic)
        count_topic = c.most_common()
        # topic을 기준으로 counting(확률 filtering해서 남은 docu 수) 후 전체 docu의 1퍼가 안넘으면 제외
        count_topic = list(filter(lambda x: (x[1] >= math.ceil(
            len(docs)*0.01) and x[0] != -1), count_topic))
        top_topic = list(map(lambda x: x[0], count_topic))
        topic_probsum = defaultdict(list)

        for t, p in topic_prob:
            if t in top_topic:
                topic_probsum[t].append(p)

        # 특정 topic으로 분류된 docu들의 확률 평균이 특정 임계값을 넘어야 함(현재는 docu_prob와 동일하게 설정)
        # 임계값을 넘은 경우 확률의 합을 score로 설정
        new_topic = []
        for top, value in topic_probsum.items():
            if sum(value)/len(value) >= mean_prob:
                new_topic.append((top, sum(value)))
                BERTopic_ticker[top].append((ticker, sum(value)))

        # concept-ticker를 설정
        if len(new_topic):
            BERTopic_concept[ticker] = new_topic
        else:
            if len(topic_probsum.items()):
                new_topic.append((list(topic_probsum.keys())[0],
                                  sum(list(topic_probsum.values())[0])))
                BERTopic_concept[ticker] = new_topic
            else:
                BERTopic_concept[ticker] = [(-1, 0)]

    for key, value in BERTopic_ticker.items():
        BERTopic_ticker[key] = sorted(value, key=lambda x: x[1], reverse=True)

    return BERTopic_concept, BERTopic_ticker


def find_similar_topic(topic_model: BERTopic, concept_n: int, sim: float, concept: str) -> list:
    """
    Using word similarity between concept and topic_embedding, find top N topics. (filter by threshold)
    if there are no topic that exceed threshold, choose a top topic from total topics

    Args:
      topic_model (BERTopic): trained BERTopic model
      concept_n (int): number of topics to find
      sim (float): similarity threshold
      concept (str): target concept to find similar topics

    Returns:
       similar_topics (list): list of similar topics

    """
    #Change some words to clear ones
    if concept=='Space':
        concept = 'Aerospace'
    elif concept == 'Tablets':
        concept = 'Tablet Computer'
    elif concept == 'Wireless Charging':
        concept = 'Inductive charging'

    #Some words ambiguous, make word less ambiguous
    similar_topics, similarity = topic_model.find_topics(
        concept, top_n=concept_n)
    # 위 결과가 상위 top_n개를 출력하기 때문에 낮은 유사도 filtering
    oversim = len(list(filter(lambda x: x >= sim, similarity)))
    pruned_similar_topics = similar_topics[:oversim]

    # filtering에 의해 모두 제외된 경우 상위 1개만 사용
    similar_topics = pruned_similar_topics if len(
        pruned_similar_topics) else [similar_topics[0]]
    return similar_topics


def zero_shot_classification(topic_model: BERTopic, concept_list: list, classifier) -> dict:
    """
        Using zero-shot classification model, each topic is classified as labels.
    Args:
        topic_model (BERTopic): BERT model that has been trained to identify topics.
        data (dict): Dictionary containing custom labels and the corresponding classifier.
        classifier: A function that takes in a string and a list of custom labels, and returns a dictionary of labels and their corresponding scores.

    Returns:
        dict: A dictionary mapping custom labels to a list of (topic, score) pairs.
    """

    last_topic = topic_model.get_topic_info().Topic.iloc[-1]
    custom_labels = copy.deepcopy(concept_list)
    for idx, label in enumerate(custom_labels):
        if label=='Space':
            custom_labels[idx] = 'Aerospace'
        if label == 'Tablets':
            custom_labels[idx] = 'Tablet Computer'
        if label == 'Wireless Charging':
            custom_labels[idx] = 'Inductive charging'

    concept_ticker = defaultdict(list)
    for t in tqdm(range(last_topic+1)):
        sequence_to_classify = " ".join(
            [word for word, _ in topic_model.get_topic(t)])
        results = classifier(sequence_to_classify, custom_labels)
        over_sim = list(
            filter(lambda x: results['scores'][x] >= 0.1, range(len(results['scores']))))
        for idx in over_sim:
            if results['labels'][idx]=='Aerospace':
                label = 'Space'
            elif results['labels'][idx]=='Tablet Computer':
                label = 'Tablet Computer'
            elif results['labels'][idx]=='Inductive charging':
                label = 'Wireless Charging'
            else:
                label = results['labels'][idx]
            concept_ticker[label].append((t, results['scores'][idx]))

        # for idx in over_sim:
        #     label = results['labels'][idx]
        #     concept_ticker[label].append((t, results['scores'][idx]))

    return concept_ticker

def concept2ticker(topic_model: BERTopic, concept_dict: dict, BERTopic_ticker: dict, concept_n: int, sim: float, classifier) -> Tuple[pd.DataFrame, defaultdict]:
    """
    Map each concept to the corresponding ticker using the BERTopic model and the zero-shot classifier.
    
    Args:
        topic_model: BERTopic object trained on a corpus of documents
        concepts: Dictionary of concepts and their corresponding tickers
        total_tic: Set of all tickers present in the documents
        BERTopic_ticker: Dictionary of topics and their corresponding tickers
        concept_n: Number of similar topics to consider for each concept
        sim: Threshold for filtering similar topics based on their similarity with the concept
        classifier: Zero-shot classifier to map topics to labels

    Returns:
        results_df: DataFrame containing the results of the mapping
        concept2news: Dictionary of concepts and their corresponding topics
        concept_path: Path where the results were saved
        scores: Tuple containing the F1 score, precision, and recall of the mapping
    """
    results_df = pd.DataFrame(
        [], columns=['concept', 'origin', 'common', 'bertopic'])

    concept_list = list(concept_dict.keys())

    concept_topic = zero_shot_classification(topic_model, concept_list, classifier)
    concept2news = defaultdict(list)
    for concept in concept_list:
        # concept의 word_embedding과 topic representation을 비교
        similar_topics = find_similar_topic(
            topic_model, concept_n, sim, concept)

        zero_shot_topics = list(map(lambda x: x[0], concept_topic[concept]))
        intersect_topics = list(set(similar_topics) | set(zero_shot_topics))
        
        if len(intersect_topics) == 0:
            intersect_topics = similar_topics

        tickers = []
        concept2news[concept].extend(intersect_topics)
        for st in intersect_topics:
            tickers.extend(BERTopic_ticker[st])

        # ticker 별로 value 재취합
        tickers_dict = defaultdict(float)
        for ticker, value in tickers:
            tickers_dict[ticker] += value

        tickers = list(filter(lambda x: x[1]>=30, tickers_dict.items()))
        tickers = list(map(lambda x: x[0], tickers))

        intersect = list(set(concept_dict.get(concept,[])) & set(tickers))

        results_df = results_df.append({"concept": concept,
                                        "origin": list(set(concept_dict.get(concept,[]))- set(intersect)),
                                        "common": intersect,
                                        "bertopic": list(set(tickers)-set(intersect))}, ignore_index=True)

    precision = cal_score(results_df['origin'], results_df['common'])
    recall = cal_score(results_df['bertopic'], results_df['common'])

    f1 = 2*((precision*recall)/(precision+recall))
    scores = (f1, precision, recall)

    concept_path = make_date_dir('./news_result')

    results_df.to_csv(os.path.join(concept_path, 'result_concept.csv'))

    return concept_path, concept2news, scores
        

def ticker2concept(topic_model: BERTopic, concepts: dict, total_tic: set, BERTopic_concept: dict, results_df: pd.DataFrame, concept_n: int, sim:float,):
    """
        Convert the concept to ticker using the BERTopic_concept dictionary.

        Args:

            topic_model: BERTopic model that has been trained on the dataset
            concepts: dictionary of concepts and their associated tickers
            total_tic: set of all tickers in the dataset
            BERTopic_concept: dictionary of tickers and their associated topics
            results_df: dataframe of the results from the concept2ticker function
            concept_n: number of topics to consider for each concept
            sim: similarity threshold for selecting similar topics

        Returns:
        
            new_result_df: dataframe of tickers and their associated concepts
        """

    new_result_df = pd.DataFrame()
    for ticker in total_tic:
        origin_concept, common_concept, bertopic_concept = [], [], []
        for i, (k, c, b) in enumerate(zip(results_df.etf, results_df.common, results_df.bertopic)):
            _concept = results_df.concept.iloc[i]
            if ticker in k:
                origin_concept.append(_concept)
            elif ticker in c:
                common_concept.append(_concept)
            elif ticker in b:
                bertopic_concept.append(_concept)
        if len(common_concept) == 0 and len(bertopic_concept) == 0:
            for concept in concepts.keys():
                similar_topics = find_similar_topic(
                    topic_model, concept_n, sim, concept)
                top_topic = list(map(lambda x: x[0], BERTopic_concept[ticker]))
                if (len(set(top_topic) & set(similar_topics))):
                    if ticker in concepts[concept]:
                        common_concept.append(concept)
                        origin_concept = list(
                            set(origin_concept) - set(common_concept))
                    if (ticker in total_tic) and (ticker not in concepts[concept]):
                        bertopic_concept.append(concept)
        new_result_df = new_result_df.append(
            {"ticker": ticker, "origin": origin_concept, "common": common_concept, "bertopic": bertopic_concept}, ignore_index=True)

    new_result_df = new_result_df.sort_values(
        by=['ticker']).reset_index(drop=True)

    return new_result_df