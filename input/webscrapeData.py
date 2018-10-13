out_dir = snakemake.params[0]

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from datetime import date, timedelta



#Function to open the driver
def open_driver():
    """
    Function to open the chrome webdriver.
    """
    # ---
    dir = out_dir+'/downloadData'
    if not os.path.exists(dir):
        os.makedirs(dir)

    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    prefs = {"download.default_directory" : dir}
    chrome_options.add_experimental_option("prefs",prefs)
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(out_dir+'/chromedriver', chrome_options=chrome_options)
    # ---
    print('Chrome driver is good to go!')
    return driver


#Function for searchterm
def build_url(start, end, region, searchterms):
    """
    Function to construct the URL.
    """
    # ---
    if isinstance(searchterms, list) == True:
        searchterms = ','.join(map(str, searchterms))
    url = "https://trends.google.ch/trends/explore?date="+start+"%20"+end+"&geo="+region+"&q="+searchterms
    # ---
    return url

    # FUNCTION

#Function download file
def download_csv(url, driver):
    """
    Function to download the csv file of the map.
    """
    # ---
    print('... start download...')

    map_dl = out_dir+'/downloadData/geoMap.csv'
    if os.path.exists(map_dl):
        os.remove(map_dl)

    export_map = []
    while len(export_map) == 0:
        print('... ... try loading the page...')
        driver.get(url)
        time.sleep(2)

        export_map = driver.find_elements_by_xpath('//*[@class="fe-geo-chart-generated fe-atoms-generic-container"]')

    export_map[0].find_element_by_xpath('.//*[@title="CSV"]').click()

    while not os.path.exists(map_dl):
        time.sleep(1)

    del export_map

    print('... download complete.')

    # ---
    return


#Function rename the file and move it
def rename_csv(start, end, region, searchterms):
    """
    Function to rename and move files.
    """
    # ---
    if isinstance(searchterms, list) == True:
        searchterms = ','.join(map(str, searchterms))
    searchterms = searchterms.replace(' ','%20')

    dir = out_dir+'/webscrapeData'
    if not os.path.exists(dir):
        os.makedirs(dir)

    map_dl = out_dir+'/downloadData/geoMap.csv'
    map_name = dir+'/map_'+searchterms+'_'+start+'_'+end+'_'+region+'.csv'

    os.rename(map_dl, map_name)

    # ---
    return

#Function to get all the dates
def get_dates():
    """
    Function to produce a list of dates with YYYY-MM-DD from 2015-11-08 to 2016-11-08.
    """
    # ---
    d1 = date(2015,8,31)
    #d1 = date(2018,8,25)
    d2 = date(2018,8,31)
    dates = [str(d1 + timedelta(days=x)) for x in range((d2-d1).days + 1)]

    # ---
    return dates

#Function download all files
def main():
    driver = open_driver()
    searchterms = 'Wohnung'
    region = 'CH'
    dates = get_dates()
    for date in dates:
        print('Download data for: ', date)
        url = build_url(date, date, region, searchterms)
        download_csv(url, driver)
        rename_csv(date, date, region, searchterms)
    driver.quit()
    print('All data downloaded.')

if __name__ == '__main__':
  main()
