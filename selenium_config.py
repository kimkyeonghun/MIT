import os
import chromedriver_autoinstaller
from selenium import webdriver


class sConfig():
    def __init__(self):
        super(sConfig, self).__init__()

    @property
    def get_config(self):
        chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[
            0]
        driver_path = f'./{chrome_ver}/chromedriver'
        if not os.path.exists(driver_path):
            chromedriver_autoinstaller.install(True)

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('headless')
        chrome_options.add_argument('window-size=1920x1080')
        chrome_options.add_argument("disable-gpu")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
        chrome_options.add_experimental_option(
            "excludeSwitches", ["enable-logging"])
        return driver_path, chrome_options
