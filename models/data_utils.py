import os
import json
from tqdm import tqdm
from collections import defaultdict

from typing import Tuple, List

import pandas as pd
import numpy as np


def load_concept() -> Tuple[dict, set]:
    with open('./concept_data/concept_list.json', 'r') as f:
        data = json.load(f)

    concept_list = []
    for _, value in data.items():
        concept_list.extend(value)

    concept_list = set(concept_list)

    return data, concept_list


def load_news_dataset() -> pd.DataFrame:

    df = pd.read_csv("./data/research_total.csv", index_col=0)
    df = df.dropna()

    return df


def load_tic2com() -> dict:
    df2 = pd.read_csv("./data/ticker2company.csv", names=[
                      'ticker', 'Name', 'Sector', 'Market Gap'], header=None).iloc[1:]
    df2 = df2.reset_index(drop=True)

    # 빠른 참조를 하기 위한 key-value 형태로 변경
    tic2com = {}
    for ticker, name in zip(df2.ticker, df2.Name):
        tic2com[ticker] = name

    return tic2com


def replace_all(content: str, dic: dict) -> str:
    for old, new in dic.items():
        content = content.replace(old, new)
    return content


def data_preprocessing(df: pd.DataFrame, tic2com: dict) -> pd.DataFrame:
    # 새 데이터셋 도입시 check 할 필요성 있음
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
    df = df.reset_index(drop=True)
    return df


def matching_news_concept(df: pd.DataFrame, topics: List[int], concept2news: defaultdict) -> pd.DataFrame:
    news_concept = []
    for topic in topics:
        if topic == -1:
            news_concept.append(np.nan)
        else:
            concept_n = ''
            for key, value in concept2news.items():
                if topic in value:
                    concept_n += key+","
            if len(concept_n):
                news_concept.append(concept_n[:-1])
            else:
                news_concept.append(np.nan)

    df['concept'] = news_concept
    n2c = df[['data_id', 'concept']]
    develop = pd.read_csv('./data/develop_total.csv', index_col=0)
    final = pd.merge(develop, n2c, how='left', on='data_id')

    return final


def get_sec_df(etf_dict: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    ETF(Kensho) 데이터와 비교하여 매칭 결과의 교집합을 찾아냄
    """
    sec_concept_df = pd.DataFrame(
        [], columns=['ticker', 'etf', 'common', 'sec'])
    for i in tqdm(range(len(df))):
        ticker, concepts = df.ticker[i], df.concept[i]
        common = [concept for concept in concepts if ticker in (
            etf_dict.get(concept, []))]
        tmp = pd.DataFrame([{
            'ticker': ticker,
            'etf': [],
            'common': common,
            'sec': list(set(concepts)-set(common))
        }])
        sec_concept_df = pd.concat([sec_concept_df, tmp], ignore_index=True)

    origins = []
    for ticker in tqdm(df.ticker):
        origin = []
        for concept in etf_dict.keys():
            if ticker in etf_dict[concept]:
                if concept not in sec_concept_df[sec_concept_df.ticker == ticker]['common'].tolist()[0]:
                    origin.append(concept)
        origins.append(origin)

    sec_concept_df['etf'] = origins

    return sec_concept_df


def cal_score(A: pd.Series, B: pd.Series) -> float:
    A_count, B_count = 0, 0
    for k, l in zip(A, B):
        B_count += len(l)
        A_count += len(k)

    return B_count/(B_count+A_count)


def save_file(path: str, template: str, data: pd.DataFrame) -> None:
    i = 0
    name = '{}_{}'.format(template, i)
    while os.path.exists(os.path.join(path, name)):
        i += 1
        name = '{}_{}'.format(template, i)
    data.to_csv("{}.csv".format(os.path.join(path, name)))
