# importing libraries
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import time
import os
import requests
from bs4 import BeautifulSoup
#---------------------------------------------------------------#

# start the chrome driver
driver = webdriver.Chrome('C:/chromedriver/chromedriver.exe')

# categories we want to search for and the url for each category
categories = {
    "soccerball": "https://unsplash.com/s/photos/soccer-ball",
    "tennisball": "https://unsplash.com/s/photos/tennis-ball",
    "golfball": "https://unsplash.com/s/photos/golf-ball",
    "basketball": "https://unsplash.com/s/photos/basket-ball",
    "poolball": "https://unsplash.com/s/photos/pool-ball"
}

# check if the image folder exists, if not create it
if not os.path.exists("images"):
    os.makedirs("images")

# loop through the categories
for category, url in categories.items():
    
    # search for the url which is the value of the category
    driver.get(url)

    images = []

    # scroll down the page and load more images until we have 150 images
    while len(images) < 150:
        driver.execute_script("window.scrollBy(0, 1000)")
        time.sleep(1)

        try:
            el = driver.find_element(By.XPATH, '//button[text()="Load more"]')
            ActionChains(driver).click(el).perform()
            time.sleep(1)
        except:
            pass

        page_html = driver.page_source

        soup = BeautifulSoup(page_html, 'html.parser')
        # find all the images on the page
        images = soup.findAll('img', {'class':"tB6UZ a5VGX"})
        
    # create a folder for each category
    category_path = os.path.join('images', category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    
    # save the images in the folder
    i = 1
    for image in images:
        try:
            response = requests.get(image['src'], stream=True)
            with open(category_path + '/'+ str(category)+str(i)+'.jpg', "wb") as file:
                file.write(response.content)
            i += 1
        except:
            pass

driver.quit()