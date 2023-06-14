import os
import re
import datetime
import subprocess
import asyncio

import pandas as pd

from log import LoggingApp
from trainer import KBERTopic


def clean_text(text):
    text = " ".join(text.split('\n'))
    pattern = r'\([^)]*\)'
    text = re.sub(pattern=pattern, repl='', string=text).strip()
    if text.find('- ') != -1:
        text = text[text.find('- ')+2:]
    elif text.find(' -- ') != -1:
        text = text[text.find(' -- ')+4:]
    elif text.find(' – ') != -1:
        text = text[text.find(' – ')+3:]
    return text


BROKERS = ['bullpen0003', 'bullpen0007', 'bullpen0008', 'bullpen0009']
password = '##########'


class BERTopicRunner(LoggingApp):
    def __init__(self,):
        super(BERTopicRunner, self).__init__()

    async def execute_ssh_command(self, broker, password):
        if broker == 'bullpen0003':
            command =  ['sshpass', '-p', password, 'ssh', f'kyeonghun.kim@{broker}', 'python', 'crawling_step.py']
        else:
            command = ['ssh', f'kyeonghun.kim@{broker}', 'python', 'crawling_step.py']

        process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=False)
        abnormal = False
        max_try_num = 10
        while process.returncode is None:
            out = await process.stdout.readline()
            #broker의 subprocess가 끝난 경우 탈출 로직
            if len(out.strip().decode('utf-8'))==0 or out.strip().decode('utf-8')==' ':
                break
            #Crawling이 잘못되었음
            if 'ERRMSG' in out.strip().decode('utf-8'):
                out = out.strip().decode('utf-8')
                idx = out.index('ERRMSG')
                out = out[idx:]
                _, _file, ticker, err_broker = out.split('|')
                process.terminate()
                abnormal = True
                break
            #Not Defined Error, Key에러와 같은 불확실한 에러인 것으로 보임
            if 'CRITICAL' in out.strip().decode('utf-8'):
                process.terminate()
                process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=False)
                max_try_num -=1
                self.send_slack("#nlp-log", 'News Crawler Bot', f"Critical Issue {broker}")
                if max_try_num:
                    continue
                else:
                    raise Exception("Critical Issue")

            self.logger.info(out.strip().decode('utf-8'))

        await process.wait()
        if abnormal:
            self.send_slack("#nlp-log", 'News Crawler Bot', f"Error {ticker} in {err_broker} about {_file}")
        else:
            self.send_slack("#nlp-log", 'News Crawler Bot', f"Finish {broker}")

    def total_results(self):
        """
        Concatenate the results of the news crawling process, clean the text data, and save the result to a CSV file.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any value.
        """
        self.logger.info("Concat files about URL data from nas")
        DATA_PATH = ''
        df = pd.DataFrame([])
        for file in os.listdir(DATA_PATH):
            if file.startswith('bullpen') and 'develop' in file:
                tmp = pd.read_csv(os.path.join(DATA_PATH, file), index_col=0)
                tmp = tmp.drop_duplicates(['data_id'])
                tmp = tmp.dropna().reset_index(drop=True)
                df = pd.concat([df, tmp])

        gather_ticker = []
        for u_id in df[df.duplicated(['data_id']) == True].data_id.unique():
            gather_ticker.append(
                (u_id, ",".join(df[df.data_id == u_id].ticker.tolist())))

        df = df.drop_duplicates(['data_id']).reset_index(drop=True)
        for d_id, tickers in gather_ticker:
            df.loc[df.data_id == d_id, 'ticker'] = tickers

        df.to_csv('./data/develop_total.csv')

        self.logger.info("Concat files about Content data from nas")
        df = pd.DataFrame([])
        for file in os.listdir(DATA_PATH):
            if file.startswith('bullpen') and 'research' in file:
                tmp = pd.read_csv(os.path.join(DATA_PATH, file), index_col=0)
                tmp = tmp.drop_duplicates(['data_id'])
                tmp = tmp.dropna().reset_index(drop=True)
                df = pd.concat([df, tmp])

        gather_ticker = []
        for u_id in df[df.duplicated(['data_id']) == True].data_id.unique():
            gather_ticker.append(
                (u_id, ",".join(df[df.data_id == u_id].ticker.tolist())))

        df = df.drop_duplicates(['data_id'])
        for d_id, tickers in gather_ticker:
            df.loc[df.data_id == d_id, 'ticker'] = tickers

        df = df.reset_index(drop=True)

        df['content'] = df['content'].apply(clean_text)
        df = df.sort_values('release_time').reset_index(drop=True)

        df.to_csv('./data/research_total.csv')

    async def main(self):
        """
            BERTopicRunner is carried out in 3 Steps
            1. Running crawling_step.py in each broker.
                It iterates over a list of brokers defined at the top of the script, and for each broker,
                it runs the crawling_step.py script on the remote machine using Secure Shell (SSH) and the
                sshpass command-line tool. The output of the script is logged to the console.

            2. Concat results
                total_results() method to concatenate the results from the different brokers into a single
                dataframe and save it to a CSV file.

            3. KBERTopic
                It instantiates a KBERTopic model, either by loading a pre-trained model if the load_mode
                flag is set to True, or by training a new model using the specified number of topics.
        """

        msg = '{}\n'.format(datetime.datetime.now().strftime('%Y %B %d. %A'))
        msg += "Crawling Start\n"
        self.send_slack("#nlp-log", 'News Crawler Bot', msg)
        tasks = []
        for broker in BROKERS:
            task = asyncio.create_task(self.execute_ssh_command(broker, password))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

    def start(self, **kwargs):
        os.chdir('/home/kyeonghun.kim/BERTopic')
        asyncio.run(self.main())
        self.send_slack("#nlp-log", 'News Crawler Bot', "Finish")
        self.load_mode = kwargs.get('load_mode', False)
        self.model_name = kwargs.get('model_name', None)
        self.reduce_topic = kwargs.get('reduce_topic', 900)
        self.total_results()
        if self.load_mode:
            model = KBERTopic(
                load_mode=True, model_name=self.model_name)
        else:
            model = KBERTopic(reduce_topic=self.reduce_topic)

        model.run()
