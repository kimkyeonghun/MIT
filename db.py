import os
import ast
import glob
import hashlib
import subprocess
import warnings

import pandas as pd

import models.db_fun as db_fun
from log import LoggingApp

warnings.filterwarnings(action='ignore')

HASH_NAME = "md5"

NAS_PATH = "/nas/DnCoPlatformDev/execution/kyeonghun.kim"

titles = {
    "concept": {
        "columns": ['concept_name', 'concept', 'updated_at', 'created_at'],
        'delete': ['concept_name', 'concept_x', 'updated_at_x', 'created_at_x'],
        'insert': ['concept_name', 'concept_y', 'updated_at_y', 'created_at_y'],
        'update': ['concept_name', 'concept_x', 'updated_at_y', 'created_at_x']
    },
    "matching": {
        "columns": ['ticker', 'concept', 'snp_company_id', 'updated_at', 'created_at'],
        'delete': ['ticker', 'concept_x', 'snp_company_id_x', 'updated_at_x', 'created_at_x'],
        'insert': ['ticker', 'concept_y', 'snp_company_id_y', 'updated_at_y', 'created_at_y'],
        'update': ['ticker', 'concept_x', 'snp_company_id_x', 'updated_at_y', 'created_at_x']
    }
}

password = 'dncodev12#'


def change_concept2hash(newly_concat_df):
    ticker_hash_table = {}
    for ticker in newly_concat_df.concept.tolist():
        text = ticker.encode('utf-8')
        md5 = hashlib.new(HASH_NAME)
        md5.update(text)
        result = md5.hexdigest()
        ticker_hash_table[ticker] = result
    return ticker_hash_table


class DBRunner(LoggingApp):
    def __init__(self,):
        super(DBRunner, self).__init__()

    def generate_KERMIT_DB(self, before_df, new_df, table_name):
        tmp = pd.merge(before_df, new_df, how='outer',
                       on=titles[table_name]['columns'][0])
        df = pd.DataFrame([], columns=titles[table_name]['columns'])
        for _, row in tmp.iterrows():
            # concept 삭제
            if row['concept_y'] != row['concept_y']:
                df = pd.concat([df,
                                pd.DataFrame(
                                    [row[titles[table_name]['delete']].values], columns=titles[table_name]['columns'])
                                ], ignore_index=True)
                continue
            # concept 추가
            if row['concept_x'] != row['concept_x']:
                df = pd.concat([df,
                                pd.DataFrame(
                                    [row[titles[table_name]['insert']].values], columns=titles[table_name]['columns'])
                                ], ignore_index=True)
                continue
            df = pd.concat([df,
                            pd.DataFrame(
                                [row[titles[table_name]['update']].values], columns=titles[table_name]['columns'])
                            ], ignore_index=True)

        return df

    def concat_sec_result(self, news_df):
        keybert_results = pd.read_csv(
            './sec_result/sec_concept.csv', index_col=0)

        newly_concat_df = pd.DataFrame(
            [], columns=['concept', 'kensho', 'common', 'kermit'])
        for concept in news_df.concept:
            bertopic_row = news_df[news_df.concept == concept]
            keybert_row = keybert_results[keybert_results.concept == concept.upper(
            )]
            common_row = set(ast.literal_eval(
                bertopic_row['common'].iloc[0])+ast.literal_eval(keybert_row['common'].iloc[0]))
            kensho_row = set(ast.literal_eval(
                bertopic_row['kensho'].iloc[0])+ast.literal_eval(keybert_row['etf'].iloc[0])) - common_row
            kermit_row = set(ast.literal_eval(bertopic_row['bertopic'].iloc[0])) & set(
                ast.literal_eval(keybert_row['sec'].iloc[0])) - common_row
            newly_concat_df = newly_concat_df.append(
                {"concept": concept,
                 'kensho': list(kensho_row),
                 'common': list(common_row),
                 'kermit': list(kermit_row)
                 }, ignore_index=True
            )
        return newly_concat_df

    def generate_tables(self, newly_concat_df):
        snp = pd.read_csv('./data/instrument_data_us_equity.csv', index_col=0)
        snp = snp.dropna(subset=['snp_companyid'])
        snp = snp.astype({'snp_companyid': 'int'})

        ticker_hash_table = change_concept2hash(newly_concat_df)

        # FIXME:다시 로드 할 때, snp_companyid가 float일수도 있음
        before_concept_list_df = pd.read_csv(
            os.path.join(NAS_PATH, 'kermit_concept.csv'))
        concept_list_df = db_fun.kermitDB_concept(ticker_hash_table)
        concept_list_df = self.generate_KERMIT_DB(
            before_concept_list_df, concept_list_df, 'concept')
        concept_list_df.to_csv(os.path.join(NAS_PATH, 'kermit_concept.csv'))

        before_concept_df = pd.read_csv(
            os.path.join(NAS_PATH, 'kermit_matching.csv'))
        concept_df = db_fun.kermitDB_mathcing(
            newly_concat_df, snp, ticker_hash_table)
        concept_df = self.generate_KERMIT_DB(
            before_concept_df, concept_df, 'matching')
        concept_df.to_csv(os.path.join(NAS_PATH, 'kermit_matching.csv'))

        # tickers = np.load('./ticker_data/ticker_list.npy')
        # ticker_dict = dict()
        # for ticker in tickers:
        #     s_id = snp[snp['ticker'] == ticker]['snp_companyid'].values
        #     if len(s_id):
        #         ticker_dict[ticker] = s_id[0]

        # news_final_df = news_final_df.sort_values(
        #     'release_time').reset_index(drop=True)
        # for i, v in enumerate(news_final_df['ticker']):
        #     new_row = []
        #     for ticker in v.split(','):
        #         if ticker_dict.get(ticker):
        #             new_row.append(str(ticker_dict[ticker]))
        #     new_row = ','.join(new_row)
        #     news_final_df.loc[i, 'ticker'] = new_row

        # news_final_df.to_csv('./data/develop_final.csv')

    def newly_news_results(self):
        news_path = os.path.join('news_result', sorted(list(
            map(lambda x: x.split("/")[-1], glob.glob('./news_result/*'))), reverse=True)[0])
        df = pd.read_csv(os.path.join(news_path, 'result_concept.csv'))
        return df

    def start(self, **kwargs):
        results_df = self.newly_news_results()
        #news_final_df = pd.read_csv()
        newly_concat_df = self.concat_sec_result(results_df)
        newly_concat_df.to_csv('./debug.csv')
        
        self.logger.info("Merge with sec results")
        self.generate_tables(newly_concat_df)

        # self.logger.info("Send kermit_concept.csv")
        # subprocess.run(['sshpass', '-p', password, 'scp', NAS_PATH + '/kermit_concept.csv',
        #                 f'jw.kim@strike0001:/home/jw.kim/uploads'])
        
        # self.logger.info("Send unimport to import")
        # result = subprocess.Popen(
        #     ['sshpass', '-p', password, 'ssh', f'jw.kim@strike0001', 'bash', 'upload_KERMIT.sh', 'kermit_concept.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        # )
        # while result.poll() == None:
        #     out = result.stdout.readline()
        #     self.logger.info(out.strip())

        self.logger.info("Send kermit_concept.csv")
        subprocess.run(['sshpass', '-p', password, 'scp', NAS_PATH + '/kermit_concept.csv',
                        f'dncodev@192.168.1.33:/home/dncodev/KERMIT'])
        # self.logger.info("Send unimport to import")
        # result = subprocess.Popen(
        #     ['sshpass', '-p', password, 'ssh', f'jw.kim@strike0001', 'bash', 'upload_KERMIT.sh', 'kermit_concept.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        # )
        # while result.poll() == None:
        #     out = result.stdout.readline()
        #     self.logger.info(out.strip())

        self.logger.info("Send kermit_matching.csv")
        subprocess.run(['sshpass', '-p', password, 'scp', NAS_PATH + '/kermit_matching.csv',
                        f'dncodev@192.168.1.33:/home/dncodev/KERMIT'])
        # self.logger.info("Send unimport to import")
        # result = subprocess.Popen(
        #     ['sshpass', '-p', password, 'ssh', f'jw.kim@strike0001', 'bash', 'upload_KERMIT.sh', 'kermit_matching.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        # )
        # while result.poll() == None:
        #     out = result.stdout.readline()
        #     self.logger.info(out.strip())
