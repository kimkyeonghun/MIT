import datetime
from collections import defaultdict
from itertools import chain

import pandas as pd

def kermitDB_concept(ticker_hash_table):
    concept_list_df = pd.DataFrame([], columns = ['concept_name', 'concept'])
    for key, value in ticker_hash_table.items():
        concept_list_df = pd.concat([
            concept_list_df, pd.DataFrame({
                "concept_name": key, 
                "concept": value
            },index=[0])
        ], ignore_index=True)

    concept_list_df['updated_at'] = datetime.datetime.now().strftime('%Y-%m-%d')
    concept_list_df['created_at'] = datetime.datetime.now().strftime('%Y-%m-%d')
    
    return concept_list_df

def kermitDB_mathcing(newly_concat_df, snp, ticker_hash_table):
    total_ticker_list = list(set(chain(*
        newly_concat_df['origin'].tolist() + newly_concat_df['common'].tolist() + newly_concat_df['kermit'].tolist()
    )))

    ticker2concept = defaultdict(list)
    for ticker in total_ticker_list:
        for _, row in newly_concat_df.iterrows():
            if ticker in row['origin'] + row['common'] + row['kermit']:
                ticker2concept[ticker].append(row['concept'])

    concept_df = pd.DataFrame([], columns = ['ticker', 'concept'])
    for key, value in ticker2concept.items():
        concept_df = pd.concat([
            concept_df, pd.DataFrame({
                "ticker":key,
                'concept': [list(map(lambda x: ticker_hash_table[x], value))]
            })
        ], ignore_index=True)

    snp_company_id = []
    for ticker in concept_df['ticker']:
        c_id = snp[snp.ticker==ticker]['snp_companyid'].values
        if len(c_id):
            snp_company_id.append(c_id[0])
        else:
            snp_company_id.append(None)
    concept_df['snp_company_id'] = snp_company_id
    concept_df['updated_at'] = datetime.datetime.now().strftime('%Y-%m-%d')
    concept_df['created_at'] = datetime.datetime.now().strftime('%Y-%m-%d')
    return concept_df