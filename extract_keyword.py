import argparse
from collections import defaultdict
import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sec_cik_mapper import StockMapper

from keybert import KeyBERT
os.chdir('/home/kyeonghun.kim/BERTopic')

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2)
parser.add_argument('--thres', type=float, )
args = parser.parse_args()

ticker_list = np.load('./ticker_list2.npy')
concept_list = pd.read_csv('./data/theme_ticker_equity.csv').theme.unique()

DATA_PATH = './edgar-crawler/datasets/EXTRACTED_FILINGS'
ITEM_LIST = ['item_1', 'item_7']


def get_description():

    with open("concept_des.txt", 'r') as f:
        data = f.readlines()

    description = defaultdict(str)

    for line in data:
        concept, des = line.strip().split("=")
        description[concept.upper()] = des

    return description


def extract_keyword(keyModel, contents):
    all_keywords = defaultdict(int)
    for content in contents:
        if len(content.split()) > 5:
            extracted_keyword = keyModel.extract_keywords(content, keyphrase_ngram_range=(1, 3), stop_words='english',
                                                          use_maxsum=True, nr_candidates=20, top_n=7)
            pruned_keyword = list(
                filter(lambda x: x[1] >= args.thres, extracted_keyword))
            for keyword in pruned_keyword:
                all_keywords[keyword[0]] += 1
    return all_keywords


def get_similarity(keyModel, keyword_embedding):
    similarity_list = []
    description = get_description()
    for concept in concept_list:
        search_embeddings = keyModel.model.embed(description[concept])
        sims = cosine_similarity(search_embeddings.reshape(
            1, -1), keyword_embedding).flatten()
        ids = np.argsort(sims)[-5:]
        similarity = [sims[i] for i in ids][::-1]
        similarity_list.append(similarity)
    assert len(concept_list) == len(similarity_list)
    return similarity_list


def main():
    keyModel = KeyBERT()
    mapper = StockMapper()

    for ticker in tqdm(ticker_list):
        print(
            f'================================{ticker}=================================')

        contents = []
        try:
            # tikcer->cik 로 변환 후 최근 N년간의 sec 10-K 파일 사용
            cik = str(int(mapper.ticker_to_cik[ticker]))
            file_lists = list(filter(lambda x: x.startswith(
                cik), os.listdir(DATA_PATH)))[-args.year:]
            for file in file_lists:
                with open(os.path.join(DATA_PATH, file)) as f:
                    data = json.load(f)
                for ITEM in ITEM_LIST:
                    contents.extend(data[ITEM].split('\n'))
        except Exception as e:
            print(e, ticker)
            continue

        all_keywords = extract_keyword(keyModel, contents)

        # TODO: 상위 N개 사용 혹은 특정 갯수 이상만 사용과 같이..
        keyword_list = list(all_keywords.keys())
        keyword_list.sort()

        if len(keyword_list) == 0 or len(file_lists) == 0:
            print(f"{ticker} No 10-K report because of foreign company")
            continue

        keyword_embedding = []
        for keyword in keyword_list:
            embeddings = keyModel.model.embed(keyword)
            keyword_embedding.append(embeddings)

        similarity_list = get_similarity(keyModel, keyword_embedding)

        text = ''
        for concept, similarity in zip(concept_list, similarity_list):
            text += f"{concept}:{similarity}\n"

        with open(f'./SEC_CTM/{ticker}_test.txt', 'w') as f:
            f.write(text)


if __name__ == '__main__':
    main()
