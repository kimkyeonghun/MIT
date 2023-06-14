from collections import defaultdict, Counter
from typing import Tuple
from tqdm import tqdm

import pandas as pd

from bertopic import BERTopic

from models.bertopic_fun import find_similar_topic

def zero_shot_classification_wiki(topic_model: BERTopic, classifier, sub_concepts: list, top_node: str) -> Tuple[dict, list]:
    """
        Using zero-shot classification model, each topic is classified as labels from sub-keyword.
        It is wikipedia version
    """
    last_topic = topic_model.get_topic_info().Topic.iloc[-1]
    custom_labels = []
    for label in sub_concepts:
        if not label.startswith(top_node):
            if label != 'Main_topic_classifications':
                custom_labels.append(label)

    concept_ticker = defaultdict(list)
    for t in tqdm(range(last_topic+1)):
        sequence_to_classify = " ".join(
            [word for word, _ in topic_model.get_topic(t)])
        results = classifier(sequence_to_classify, custom_labels)
        over_sim = list(
            filter(lambda x: results['scores'][x] >= 0.1, range(len(results['scores']))))
        for idx in over_sim:
            label = results['labels'][idx]
            concept_ticker[label].append((t, results['scores'][idx]))

    return concept_ticker, custom_labels


def subconcept2ticker(topic_model, top_node, sub_concept, sub_tickers, classifier, BERTopic_ticker, concept_n, sim, ticker_n):
    """
        Intersect zero_shot classification's results and BERTopic's results about sub_concept--> (concept-topic) 
        Using BERTopic_ticker(topic-tikcer), matching concept-ticker(concept-topic-ticker)
    """
    results_df2 = pd.DataFrame([], columns=['sub-concept', 'instruments'])
    concept_topic, pruned_concept = zero_shot_classification_wiki(
        topic_model, classifier, sub_concept, top_node)

    for concept in pruned_concept:
        similar_topics = find_similar_topic(
            topic_model, concept_n, sim, concept)

        zero_shot_topics = list(map(lambda x: x[0], concept_topic[concept]))

        intersect_topics = list(set(similar_topics) & set(zero_shot_topics))
        if len(intersect_topics) == 0:
            intersect_topics = similar_topics
        intersect, tickers = [], []
        for st in intersect_topics:
            tickers.extend(BERTopic_ticker[st])

        tickers_dict = defaultdict(float)
        for ticker, value in tickers:
            tickers_dict[ticker] += value

        tickers = sorted(tickers_dict.items(),
                         key=lambda x: x[1], reverse=True)[:ticker_n]
        tickers = list(map(lambda x: x[0], tickers))

        results_df2 = results_df2.append({'sub-concept': concept,
                                          'instruments': list(set(sub_tickers) & set(tickers))}, ignore_index=True)

        return results_df2
