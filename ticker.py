import os
import re
import json
import time
import datetime
import subprocess
import warnings
from typing import Tuple, List

import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from log import LoggingApp
from selenium_config import sConfig

from keyword_extraction import KeyBERTRunner

warnings.filterwarnings(action='ignore')

BROKER_FILE_DICT = {'ticker_list_0.txt': 'bullpen0003', 'ticker_list_1.txt': 'bullpen0007',
                    'ticker_list_2.txt': 'bullpen0008', 'ticker_list_3.txt': 'bullpen0009'}

BROKERS = ['bullpen0003', 'bullpen0007', 'bullpen0008', 'bullpen0009']
password = ''


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


class TickerRunner(LoggingApp):
    def __init__(self):
        os.chdir("/home/kyeonghun.kim/BERTopic")
        super(TickerRunner, self).__init__()
        self.driver_path, self.chrome_options = sConfig().get_config
        self.logger.info(f"Chrome Driver Path is: {self.driver_path}")
        self.ticker_list: np.ndarray = np.load('ticker_data/ticker_list.npy')

    def crawling_nasdaq_screener(self):
        self.logger.info("Crawling nasdaq screener using selenium")
        driver = webdriver.Chrome(
            self.driver_path, options=self.chrome_options)
        driver.get("https://www.nasdaq.com/market-activity/stocks/screener")
        download = driver.find_element(
            by=By.XPATH, value='/html/body/div[2]/div/main/div[2]/article/div[3]/div[1]/div/div/div[3]/div[2]/div[2]/div/button')
        driver.implicitly_wait(5)
        download.send_keys(Keys.ENTER)
        driver.implicitly_wait(5)
        time.sleep(10)
        driver.quit()
        self.logger.info("Finish crawling nasdaq screener using selenium")

    def compare_ticker_pool(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
            This function takes a dataframe containing a list of stock ticker information from Nasdaq Screener as an input.
            It compares the data in the input dataframe with the ticker_list attribute
            (which is a numpy array of stock tickers that is already available to the TickerRunner class) to identify newly
            added and removed tickers.

            Inputs:

                df (pandas dataframe): A dataframe containing stock ticker information from Nasdaq Screener.

            Outputs:

                in_ticker (list of str):
                    A list of newly added stock tickers that meet the following criteria:
                        their market cap is in the top 75th percentile of all the stocks in the ticker_list attribute
                        their volume is in the top 75th percentile of all the stocks in the ticker_list attribute

                out_ticker (list of str):
                    A list of stock tickers that will be removed from the ticker_list attribute
                    because their last sale price is less than or equal to $10.0.
        """

        # filter the dataframe to include only tickers that are in the existing ticker list
        ticker_info = df[df['Symbol'].isin(self.ticker_list)]

        # find the 75th percentile of market caps in the existing ticker list
        market_cap_q3 = ticker_info['Market Cap'].quantile(.75)

        # find the new tickers that have a market cap greater than or equal to the 75th percentile
        # and remove any tickers that are already in the existing ticker list
        market_cap_new_ticker = set(df[df['Market Cap'] >= market_cap_q3].Symbol.tolist(
        )) - set(ticker_info.Symbol.tolist())

        # find the 75th percentile of volumes in the existing ticker list
        volume_q3 = ticker_info['Volume'].quantile(.75)
        # find the new tickers that have a volume greater than or equal to the 75th percentile
        # and remove any tickers that are already in the existing ticker list
        volume_new_ticker = set(
            df[df['Volume'] >= volume_q3].Symbol.tolist()) - set(ticker_info.Symbol.tolist())

        # find the intersection of the two sets of new tickers (those with high market caps and high volumes)
        in_ticker = list(market_cap_new_ticker & volume_new_ticker)

        # remove the ticker 'GOOG' from the list of new tickers
        in_ticker.remove('GOOG')
        self.logger.info(f"Newly added tickers: {in_ticker}")

        # convert the 'Last Sale' column to a numeric type and find the tickers with a last sale price less than or equal to 10
        ticker_info['Last Sale'] = ticker_info['Last Sale'].apply(
            lambda x: float(x[1:]))
        out_ticker = ticker_info[ticker_info['Last Sale']
                                 <= 10.0].Symbol.tolist()
        self.logger.info(f"Will delete tickers: {out_ticker}")

        remove_ticker = list(set(self.ticker_list) - set(df['Symbol']))
        self.logger.info(f"Will remove tickers: {remove_ticker}")

        # return the lists of new and deleted tickers
        return in_ticker, out_ticker, remove_ticker

    def update_ticker2url(self, in_ticker: List[str]) -> None:
        """
        Update the ticker-to-URL mapping using the newly added tickers.

        Args:
            in_ticker (List[str]): The list of new tickers obtained from the previous step.

        Returns:
            None: The method does not return any value.

        """
        with open('./ticker_data/ticker2url.json', 'r') as f:
            hrefs: dict = json.load(f)

        self.logger.info("Update Ticker2URL json file")

        for ticker in in_ticker:
            try:
                driver = webdriver.Chrome(
                    self.driver_path, options=self.chrome_options)

                investing_url = "https://www.investing.com"
                driver.get(investing_url)

                driver.implicitly_wait(5)
                search = driver.find_element(
                    by=By.XPATH, value='/html/body/div[5]/header/div[1]/div/div[3]/div[1]/input')

                driver.implicitly_wait(5)
                search.send_keys(ticker)
                rows = driver.find_element(
                    by=By.XPATH, value='html/body/div[5]/header/div[1]/div/div[3]/div[2]/div[1]/div[1]/div[2]/div')

                for row in rows.find_elements(by=By.TAG_NAME, value='a'):
                    self.logger.info(row.text)
                    if 'USA' in row.find_element(by=By.TAG_NAME, value='i').get_attribute('class'):
                        href: str = row.get_attribute('href')
                        self.logger.info(href)
                        row.click()
                        break
                time.sleep(6)
                hrefs[ticker] = href+'-news'
                driver.quit()
            except:
                pass

        self.logger.info("Save New Ticker2URL json file")
        with open('./ticker_data/ticker2url.json', 'w') as f:
            json.dump(hrefs, f)

    def split_ticker2broker(self, max_num: int, in_ticker: List[str]) -> None:
        """
        Split the updated ticker list into smaller groups and send each group to a different broker for further processing.

        Args:
            max_num (int): The maximum number of tickers that can be assigned to each broker.
            in_ticker (List[str]): The list of new tickers obtained from the previous step.

        Returns:
            None: The method does not return any value.

        """
        self.logger.info("Starting split ticker pools and send each broker")
        for broker_file in BROKER_FILE_DICT.keys():
            with open(os.path.join('ticker_data', broker_file), 'r') as f:
                broker = list(map(lambda x: x.strip(), f.readlines()))

            broker_out_ticker = set(broker) - set(self.ticker_list)
            for ticker in broker_out_ticker:
                broker.remove(ticker)

            diff = max_num - len(broker)
            while diff and len(in_ticker):
                broker.append(in_ticker.pop())
                diff -= 1

            to_broker = BROKER_FILE_DICT[broker_file]
            with open(os.path.join('ticker_data', broker_file), 'w') as f:
                f.write("\n".join(broker))

            if to_broker != 'bullpen0003':
                subprocess.run(
                    ['scp', os.path.join('ticker_data', broker_file), f'kyeonghun.kim@{to_broker}:/home/kyeonghun.kim/investing'])
                subprocess.run(
                    ['scp', './ticker_data/ticker2url.json', f'kyeonghun.kim@{to_broker}:/home/kyeonghun.kim/investing'])
            else:
                subprocess.run(['sshpass', '-p' f'{password}', 'scp', './ticker_data/ticker2url.json',
                                f'kyeonghun.kim@{to_broker}:/home/kyeonghun.kim/investing'])
                subprocess.run(['sshpass', '-p' f'{password}', 'scp', os.path.join('ticker_data', broker_file),
                                f'kyeonghun.kim@{to_broker}:/home/kyeonghun.kim/investing'])

            self.logger.info(
                f"Send File {os.path.join('ticker_data',broker_file)}, ticker2url.json to {to_broker}")

    def start(self, **kwargs):
        """
            TickerRunner is carried out in 4 Steps
                1. Crawling Nasdaq Screener:
                    The first step of the TickerRunner class is to crawl the Nasdaq website for the latest
                    stock information. This is done using the Selenium web scraping library. The script
                    navigates to the Nasdaq screener page, clicks on the download button to download the 
                    latest stock information, and then saves the downloaded data to a file.

                2. Compare ticker pool using Nasdaq Screener:
                    In the second step, the script compares the list of ticker symbols obtained from the crawl
                    with a pre-existing list of ticker symbols. If there are any new tickers that meet certain
                    criteria, such as a minimum market capitalization and volume, they are added to the list of
                    tickers. The script also identifies any tickers that no longer meet the criteria and logs them.

                3. Update Ticker to URL:
                    If there are any new tickers that were added to the list in the previous step, the script updates
                    a JSON file with the new tickers and their corresponding URLs. This is done by using the Selenium
                    web driver to navigate to the Investing.com website, search for each new ticker, and extract its URL.

                4. Split Ticker pool and Send to each broker:
                    In the final step, the script splits the updated list of tickers into smaller groups and sends each group
                    to a different broker for further processing. This is done by using the subprocess module to run separate
                    scripts for each broker, passing the relevant ticker list as an argument to each script.

        """
        # Step1
        self.crawling_nasdaq_screener()
        filename = list(filter(lambda x: x.startswith(
            'nasdaq_screener'), os.listdir()))[0]
        self.logger.info(f"Nasdaq_screener Filename is {filename}")
        df = pd.read_csv(filename)
        os.remove(filename)

        # Step2
        in_ticker, out_ticker, remove_ticker = self.compare_ticker_pool(df)
        if len(in_ticker) + len(out_ticker) + len(remove_ticker) == 0:
            self.logger.info("There are no change in Ticker Pool")
            return

        # Step3
        if len(in_ticker):
            self.update_ticker2url(in_ticker)
            key = KeyBERTRunner()
            key.matching_concept(in_ticker)
            key.pruning_results()

        # newly add ticker by rule
        for it in in_ticker:
            self.ticker_list = np.append(self.ticker_list, it)

        # newly add ticker by rule
        for ot in out_ticker:
            self.ticker_list = np.delete(
                self.ticker_list, np.where(self.ticker_list == ot))

        # ticker change or delisting
        for rt in remove_ticker:
            self.ticker_list = np.delete(
                self.ticker_list, np.where(self.ticker_list == rt))

        max_num = int(np.ceil(len(self.ticker_list)/4))
        self.logger.info(f"Number of Ticker in Each Broker: {max_num}")
        np.save('./ticker_data/ticker_list.npy', self.ticker_list)

        # Step4
        self.split_ticker2broker(max_num, in_ticker)
