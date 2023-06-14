from transformers import pipeline
from collections import defaultdict
from collections import Counter, defaultdict
import json
import math
import os
from tqdm import tqdm
import warnings

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from typing import Tuple
from umap import UMAP

# BERTopic fit 과정에서 warning을 제거하기 위해
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
warnings.filterwarnings(action='ignore')


def cal_score(A, B):
    A_count, B_count = 0, 0
    for k, l in zip(A, B):
        B_count += len(l)
        A_count += len(k)

    return B_count/(B_count+A_count)


def load_kensho():
    with open('./concept_list.json', 'r') as f:
        data = json.load(f)

    kensho_list = []
    for _, value in data.items():
        kensho_list.extend(value)

    kensho_list = set(kensho_list)

    return data, kensho_list


def load_news_dataset(kensho_list, only_news=False, intersect=False):
    df = pd.read_csv("./data/us_equities_news_dataset.csv")
    df = df.dropna()

    # news 기사만 선택적으로 사용하는 경우
    if only_news:
        df = df[df.category == 'news']
    df = df.reset_index(drop=True)

    if intersect:
        # kensho_list와 교집합임 ticker만을 새로 분류
        new_df = pd.DataFrame([])
        for ticker in list(kensho_list & set(df.ticker.unique())):
            if len(new_df):
                new_df = pd.concat([new_df, df[df.ticker == ticker]])
            else:
                new_df = df[df.ticker == ticker]
        df = new_df

    return df


def load_news_dataset2():
    df = pd.read_csv('./data/research_total.csv', index_col=0)
    df = df.dropna()

    tickers = df.ticker.apply(lambda x: x.split(',')).tolist()
    total_tic = []
    for row in tickers:
        total_tic.extend(row)

    total_tic = set(total_tic)

    return df, total_tic


def load_tic2com():
    df2 = pd.read_csv("ticker2company.csv", names=[
                      'ticker', 'Name', 'Sector', 'Market Gap'], header=None).iloc[1:]
    df2 = df2.reset_index(drop=True)

    # 빠른 참조를 하기 위한 key-value 형태로 변경
    tic2com = {}
    for ticker, name in zip(df2.ticker, df2.Name):
        tic2com[ticker] = name

    return tic2com


def replace_all(content, dic):
    for old, new in dic.items():
        content = content.replace(old, new)
    return content


def data_preprocessing(df, tic2com):
    single_chr = ['s', 'p', 'm', 'M', 'e', 'l', 'f']
    replace_dic = {"nyse": "", 'nasdaq': "", 'nysemkt': "", 'stocks': '', 'stock': '',
                   'common': '', 'shars': '', 'sales': '', 'zacks': '', 'investments': '',
                   'inc.': '', 'inc': '', 'company': '', 'zacks investment research': '',
                   'investing com': '', 'seeking alpha': '', 'bloomberg': '', 'the moytley fool': '', 'nicholas santiago': ''
                   }
    new_contents = []
    for content in tqdm(df.content):
        link = False
        new_content = []
        words = content.split()
        for word in words:
            if word in single_chr:
                continue
            if link:
                link = False
                continue
            # ticker를 company name으로 변경
            if tic2com.get(word):
                word = tic2com[word]
            # NYSE APPL 과 같이 링크 흔적이 남아있음
            if word == 'NYSE' or word == 'NASDAQ' or word == 'NYSEMKT':
                link = True
            new_content.append(word)
        test_news = " ".join(new_content)
        test_news = replace_all(test_news, replace_dic)
        new_contents.append(test_news)

    df['content'] = new_contents
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def load_BERTopic_model():
    return BERTopic.load("RawOnlyNews")


def filtering_tic_and_com(topics: list, probs: list, concept_list: set, df: pd.DataFrame, total_tic: set, docu_prob: float, mean_prob=None) -> Tuple[dict, dict]:
    """
        Using fitted BERTopic model, transform dataset and get (topic, prob) pairs.

        Filtering: if one docu's prob is lower than docu_prob, it can be interpreted uncertain topic.
        Gathering: After counting by topic, if number of docus is small, it might be outlier topic.
        Matching: match BERTopic_concept(ticker-topics) and BERTopic_ticker(topic-tickers)
    """
    if mean_prob == None:
        mean_prob = docu_prob

    BERTopic_ticker, BERTopic_concept = defaultdict(list), defaultdict(list)
    for ticker in list(concept_list & set(total_tic)):
        docs = df[df['ticker'].apply(
            lambda x: ticker in x.split(','))].content.tolist()
        sub_topics = [topics[i] for i in df[df['ticker'].apply(
            lambda x: ticker in x.split(','))].index]
        sub_probs = [probs[i] for i in df[df['ticker'].apply(
            lambda x: ticker in x.split(','))].index]
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


def find_similar_topic(topic_model, concept_n, sim, concept):
    similar_topics, similarity = topic_model.find_topics(
        concept, top_n=concept_n)
    # 위 결과가 상위 top_n개를 출력하기 때문에 낮은 유사도 filtering
    oversim = len(list(filter(lambda x: x >= sim, similarity)))
    pruned_similar_topics = similar_topics[:oversim]

    # filtering에 의해 모두 제외된 경우 상위 1개만 사용
    similar_topics = pruned_similar_topics if len(
        pruned_similar_topics) else [similar_topics[0]]
    return similar_topics


classifier = pipeline('zero-shot-classification',
                      model='facebook/bart-large-mnli', device=0)


def zero_shot_classification(topic_model, etf, classifier):
    last_topic = topic_model.get_topic_info().Topic.iloc[-1]
    custom_labels = list(etf.keys())

    BERTopic_concept = defaultdict(list)
    for t in tqdm(range(last_topic+1)):
        sequence_to_classify = " ".join(
            [word for word, _ in topic_model.get_topic(t)])
        results = classifier(sequence_to_classify, custom_labels)
        over_sim = list(
            filter(lambda x: results['scores'][x] >= 0.1, range(len(results['scores']))))
        for idx in over_sim:
            label = results['labels'][idx]
            BERTopic_concept[label].append((t, results['scores'][idx]))

    return BERTopic_concept


def concept2ticker(topic_model: BERTopic, concepts: dict, BERTopic_ticker: dict, total_tic: set, concept_n: int, sim: float, classifier):
    """
        Intersect zero_shot classification's results and BERTopic's results. --> (concept-topic) 
        Using BERTopic_ticker(topic-tikcer), matching concept-ticker(concept-topic-ticker)
    """
    results_df = pd.DataFrame(
        [], columns=['concept', 'etf', 'common', 'bertopic'])

    concept_topic = zero_shot_classification(topic_model, concepts, classifier)
    concept2news = defaultdict(list)
    for concept in concepts.keys():
        # concept의 word_embedding과 topic representation을 비교
        similar_topics = find_similar_topic(
            topic_model, concept_n, sim, concept)

        zero_shot_topics = list(map(lambda x: x[0], concept_topic[concept]))
        intersect_topics = list(set(similar_topics) | set(zero_shot_topics))

        if len(intersect_topics) == 0:
            intersect_topics = similar_topics

        intersect, tickers = [], []
        concept2news[concept].extend(intersect_topics)
        for st in intersect_topics:
            tickers.extend(BERTopic_ticker[st])

        # ticker 별로 value 재취합
        tickers_dict = defaultdict(float)
        for ticker, value in tickers:
            tickers_dict[ticker] += value

        # tickers = sorted(tickers_dict.items(),
        #                  key=lambda x: x[1], reverse=True)[:ticker_n]
        tickers = list(filter(lambda x: x[1] >= 30, tickers_dict.items()))
        tickers = list(map(lambda x: x[0], tickers))
        intersect = list(set(concepts[concept]) & set(tickers))

        results_df = results_df.append({"concept": concept,
                                        "etf": list(((set(concepts[concept]) - set(intersect)) & total_tic)),
                                        "common": intersect,
                                        "bertopic": list(set(tickers)-set(intersect))}, ignore_index=True)

    precision = cal_score(results_df['etf'], results_df['common'])
    recall = cal_score(results_df['bertopic'], results_df['common'])

    f1 = 2*((precision*recall)/(precision+recall))
    scores = (f1, precision, recall)
    print(
        f"F1-score: {scores[0]}, Precision: {scores[1]}, Recall: {scores[2]}")

    return results_df, scores


def ticker2concept(kensho, topic_model, df, BERTopic_concept, results_df, concept_n, sim,):
    kensho_list = []
    for _, value in kensho.items():
        kensho_list.extend(value)

    kensho_list = set(kensho_list)

    new_result_df = pd.DataFrame()
    for ticker in (set(df.ticker.unique()) & kensho_list):
        kensho_concept, common_concept, bertopic_concept = [], [], []
        for i, (k, c, b) in enumerate(zip(results_df.kensho, results_df.common, results_df.bertopic)):
            if ticker in k:
                kensho_concept.append(results_df.Concept.iloc[i])
            elif ticker in c:
                common_concept.append(results_df.Concept.iloc[i])
            elif ticker in b:
                bertopic_concept.append(results_df.Concept.iloc[i])
        if len(common_concept) == 0 and len(bertopic_concept) == 0:
            for concept in kensho.keys():
                similar_topics = find_similar_topic(
                    topic_model, concept_n, sim, concept)
                top_topic = list(map(lambda x: x[0], BERTopic_concept[ticker]))
                if (len(set(top_topic) & set(similar_topics))):
                    if ticker in kensho[concept]:
                        common_concept.append(concept)
                        kensho_concept = list(
                            set(kensho_concept) - set(common_concept))
                    if (ticker in kensho_list) and (ticker not in kensho[concept]):
                        bertopic_concept.append(concept)
        new_result_df = new_result_df.append(
            {"ticker": ticker, "kensho": kensho_concept, "common": common_concept, "bertopic": bertopic_concept}, ignore_index=True)

    new_result_df = new_result_df.sort_values(
        by=['ticker']).reset_index(drop=True)
    # new_result_df.to_csv('./results_ticker_df.csv')
    return new_result_df

# ETF ver


def load_etf():
    etf_df = pd.read_csv('./data/theme_ticker_equity.csv')
    ETF_concept = [string.title() for string in etf_df.theme.unique()]
    ETF_concept = set(ETF_concept)
    answer = dict()
    etf_list = []
    for kl in ETF_concept:
        value = list(etf_df[etf_df.theme == kl.upper()].equity)
        answer[kl] = value
        etf_list.extend(value)
    etf_list = set(etf_list)
    return answer, etf_list


def main():
    print("Load ETF")
    data, concept_list = load_etf()
    print("Load News")
    df, total_tic = load_news_dataset2()
    print("Cleaning News")
    tic2com = load_tic2com()
    df = data_preprocessing(df, tic2com)
    docs = df['content'].tolist()

    # HDBSCAN, UMAP 튜닝 가능
    umap_model = UMAP(n_neighbors=15,
                      n_components=5, min_dist=0.0, metric='cosine')
    vectorizer_model = CountVectorizer(
        stop_words='english', ngram_range=(1, 3), min_df=20)

    maximum_topic = 2000
    new_topic = 150
    results = []
    print("Finding start")
    while new_topic < maximum_topic:
        topic_model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer_model,
                               verbose=False, diversity=0.8, nr_topics=new_topic)
        topics, probs = topic_model.fit_transform(docs)
        # need checking
        BERTopic_concept, BERTopic_ticker = filtering_tic_and_com(
            topics, probs, concept_list, df, total_tic, 0.4)
        results_df, score = concept2ticker(
            topic_model, data, BERTopic_ticker, total_tic, 10, 0.35, classifier)
        #ticker_df = ticker2concept(answer, topic_model, df, BERTopic_concept, results_df, 5, 0.375)
        print(
            f"[Topic Number: {new_topic}] F1-score: {score[0]}, Precision: {score[1]}, Recall: {score[2]}")
        results.append([new_topic, score[0], score[1], score[2]])
        new_topic += 50

    correct_df = pd.DataFrame(results)
    correct_df.to_csv("./results_total_correct.csv")


if __name__ == '__main__':
    main()
