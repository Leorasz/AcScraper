import os
import re
import json
import torch
import ollama   
import threading
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["clean_up_tokenization_spaces"] = "False"

"""
TODO:
-Locate retardation
"""
def talk(system, prompt):
    messages = [
        # {'role': 'system','content': system},
        # {'role': 'user','content': "Use this data to find me the President of State University and make sure to end your response with RES: and then your answer: ***We are proud to welcome our new president, John M. Banks. He steps into the shoes of a decades-old tradition of excellence here at State University, and we are certain he can fill them.***"},
        # {'role': 'system','content': system},
        # {'role':'assistant', 'content':"The President of State University, at least from this text data, seems to be John M. Banks. This is clearly stated in the beginning of the text, and it further implies that he is the new president. RES: John M. Banks"},
        {'role': 'system','content': system},
        {'role': 'user','content': prompt}
    ]
    response = ollama.chat(model='llama3', messages=messages)
    return response['message']['content']

dev = "mps"

model = SentenceTransformer('all-MiniLM-L6-v2')

def findRES(stri):
    match = re.search("RES(.*)", stri)
    payload = match.group(1)
    return payload

system = "You are an expert at extracting information from text data, which is given to you surrounded by ***. Always respond with the information you've extracted in JSON format."
system_base = "You are a helpful assistant, willing to do whatever the user asks you."
def find_person(college, person):
    with sync_playwright() as p:
        try:
            print(f"Looking at person {person} for college {college}")
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("https://www.google.com/")
            # print("Google launched")
            page.wait_for_load_state('load')
            page.fill('#APjFqb', college + " " + person)
            page.press('#APjFqb', 'Enter')
            # print("Google searched")

            page.wait_for_selector('h3')
            search_results = page.query_selector_all('h3')
            # print("Searching titles for best one")
            titles = [result.inner_text() for result in search_results]
            soup = BeautifulSoup(page.content(), 'html.parser')
            form1 = soup.find('div', {'class': 'PZPZlf ssJ7i B5dxMb'})
            form2 = soup.find('div', {'class': 'FLP8od'})
            form3 = soup.find('div', {'class': 'IZ6rdc'})
            forms = [form1, form2, form3]
            possible_names = [form.text for form in forms if form]
            if possible_names and "$" not in possible_names[0]:
                resses[-1] = {"College":college, "Position":"President", "Name":possible_names[0], "Method":"Init"}
                print(possible_names[0])
                return

            target = college + "Office of " + person
            tens = torch.tensor(model.encode([target] + titles)).to(dev)
            tens /= torch.norm(tens, dim=1, keepdim=True)
            sims = tens[1:] @ tens[0]
            best_pos = torch.argmax(sims).item()
            # print(f"Clicking on best title, which is '{titles[best_pos]}'")
            search_results[best_pos].click()

            page.wait_for_load_state('load')
            # print("Page loaded")
            page_content = page.content()        
            soup = BeautifulSoup(page_content, 'html.parser')
            text_data = soup.get_text()
            prompt = f"Here's some data, tell me about the {person} of {college}: ***{text_data}***"
            # print(f"Asking Llama to extract information from: {text_data}")
            raw_out = talk(system_base, prompt)
            prompt_data = f"Who is the {person} of {college}? End your response with RES and then your answer. If you don't know, then say 'idk'. Here's some data to help you out: ***{titles[best_pos]}: {raw_out}***"
            refined_out = talk(system_base, prompt_data)
            # print(f"Raw output is: {raw_out}")
            payload = findRES(refined_out)
            # print("Information extracted")
            browser.close()
            resses[-1] = {"College":college, "Position":"President", "Name":payload, "Title":titles[best_pos], "Raw Out":raw_out, "Refined Out":refined_out, "Method":"Full ask"}
            print(payload)
        except Exception as e:
            resses[-1] = {"College":college, "Position":"President", "Name":"ERROR", "Exception":str(e), "Method":"None"}
            print("ERROR")

with open("name_wiki_link.json", "r") as file:
    data = json.load(file)

colleges = [i["Name"] for i in data]
resses = []
js = []
for college in tqdm(colleges):
    resses.append(0)
    thread = threading.Thread(target=find_person, args=(college, "President"))
    thread.start()
    thread.join()
with open("presidents_verbose2.json", "w") as file:
    json.dump(resses, file, indent=4)