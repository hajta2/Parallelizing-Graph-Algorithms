import sys, getopt
import time
from selenium import webdriver


def main(argv):
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/x-tar")
    input_link = ''
    try:
        opts, args = getopt.getopt(argv,"i:",["link="])
    except getopt.GetoptError:
        print ('-i <input_link>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--link"):
            input_link = arg
    try:
        browser=webdriver.Firefox(firefox_profile=profile, executable_path='/home/hajta2/Downloads/geckodriver')
        browser.get(input_link)
        element=browser.find_element_by_link_text("Matrix Market")
        print(element)
        element.click()
        time.sleep(10)
        browser.close()
    except:
        print("Invalid URL")
        browser.close()
        
if __name__ == "__main__":
    main(sys.argv[1:])