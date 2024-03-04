import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from tqdm import tqdm

# Hyperparameters
main_uri = 'http://www.famousquotesandauthors.com'


# ================

def scrape_index():
    quote_index = requests.get('http://www.famousquotesandauthors.com/quotes_by_topic.html')
    soup = BeautifulSoup(quote_index.content, "html.parser")
    # Scrape all uri from the tag and clean anything that doesn't conform to format /topics/x.html
    index_uris = [uri['href'] if not (uri['href'].startswith(main_uri)) else uri['href'][len(main_uri):] for uri in
                  soup.find_all("a", href=re.compile("/topics"))]
    if np.all([uri.startswith(main_uri) for uri in index_uris], where=[False]):
        print("Data is clean from unwanted prefix")
    else:
        print("Data is unclean")
    return index_uris


scraped_index = scrape_index()


def format_string(input_list):
    formatted_list = []
    for s in input_list:
        # Replace escaped single quotes (e.g., '\'') with a regular single quote ('')
        s = re.sub("\\'", "'", s)
        # Remove unwanted characters (e.g., line breaks, spaces, and extra double quotes)
        s = re.sub(r"[^\w\s]", '', s)
        # Remove leading/trailing whitespace and convert multiple consecutive whitespace characters to a single space
        s = re.sub(r"\s+", ' ', s).strip()
        formatted_list.append(s)
    return formatted_list


def scrape_quotes(uri: str):
    list_of_quotes = requests.get(f"{main_uri}{uri}")
    soup = BeautifulSoup(list_of_quotes.content, 'html.parser')
    quotes = [q.string for q in soup.find_all("div", style="font-size:12px;font-family:Arial;")]
    return format_string(quotes)

print("Scraping Started")
quotes = []
for scraped_quote in [scrape_quotes(uri) for uri in scraped_index]:
    quotes = [*quotes, *scraped_quote]
print("Scraping Finished")

print("Saving Started")
quote_dataset = "\n\n".join(quotes)
with open('q_datasets.txt', 'w') as f:
    f.write(quote_dataset)
print("Saving Finished")
