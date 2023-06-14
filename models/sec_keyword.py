from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

import os
import json

from sec_cik_mapper import StockMapper
from sklearn.metrics.pairwise import cosine_similarity

from keybert import KeyBERT

df = pd.read_csv('./nasdaq_screener_1655180302707.csv')
nasdaq_top_200 = df.sort_values('Market Cap', ascending=False)[:200]

#keyModel = KeyBERT(model="nlpaueb/sec-bert-shape")
keyModel = KeyBERT()
mapper = StockMapper()

DATA_PATH = './edgar-crawler/datasets/EXTRACTED_FILINGS'
item_list = ['item_1', 'item_1A', 'item_1B','item_7', 'item_10']

concept = pd.read_csv('./data/theme_ticker_equity.csv')
concept_list = concept.theme.unique()

for ticker in nasdaq_top_200['Symbol'].tolist():
    print(f'================================{ticker}=================================')
    all_keywords = defaultdict(int)
    try:
        cik = str(int(mapper.ticker_to_cik[ticker]))
        file_lists = list(filter(lambda x: x.startswith(cik), os.listdir('./edgar-crawler/datasets/EXTRACTED_FILINGS/')))[-2:]
        contents = []
        for file in file_lists:
            with open(os.path.join(DATA_PATH, file)) as f:
                data = json.load(f)
            for item in item_list:
                contents.extend(data[item].split('\n'))
        for content in tqdm(contents):
            if len(content.split())>5:
                keywords = list(filter(lambda x: x[1]>=0.3, keyModel.extract_keywords(content, keyphrase_ngram_range=(1, 4), stop_words='english',
                               use_maxsum=True, nr_candidates=15, top_n=5)))
                for keyword in keywords:
                    all_keywords[keyword[0]]+=1
    except Exception as e:
        print(e)
        print(ticker)
        continue
    keyword_list = list(all_keywords.keys())
    keyword_list.sort()

    if len(keyword_list)==0 and len(file_lists)==0:
        print(f"{ticker} No 10-K report because after 2020 or foreign company")
        continue

    keyword_embedding = []
    for keyword in tqdm(keyword_list):
        embeddings = keyModel.model.embed(keyword.split())
        keyword_embedding.append(np.average(embeddings, axis=0)*all_keywords[keyword])    
    similarity_list = []
    for concept in concept_list:
        search_embeddings = keyModel.model.embed([concept.lower()])
        sims = cosine_similarity(search_embeddings.reshape(1,-1), keyword_embedding).flatten()
        ids = np.argsort(sims)[-10:]
        similarity = [sims[i] for i in ids][::-1]
        similar_keyword = [keyword_list[idx] for idx in ids][::-1]
        similarity_list.append(similarity)
    assert len(concept_list)==len(similarity_list)
    text = ''
    for concept, similarity in zip(concept_list, similarity_list):
        text += f"{concept}:{similarity}\n"

    with open(f'./SEC_CTM/{ticker}_new.txt', 'w') as f:
        f.write(text)