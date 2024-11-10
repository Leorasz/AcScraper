import re
import torch
import ollama
import asyncio
from bs4 import BeautifulSoup
from transformers import pipeline
from playwright.async_api import async_playwright
from sentence_transformers import SentenceTransformer

dev = "mps"
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="tanliboy/lambda-qwen2.5-14b-dpo-test", max_new_tokens=100, device="mps:0")
# print(pipe(messages))

model = SentenceTransformer('all-MiniLM-L6-v2')

def w():
    _ = input(":")

async def search(page, query):
    await page.fill('#APjFqb', query)
    await page.press('#APjFqb', 'Enter')
    await page.wait_for_load_state('load')
    search_results = await page.query_selector_all('h3')
    titles = [await result.inner_text() for result in search_results]
    soup = BeautifulSoup(await page.content(), 'html.parser')
    form1 = soup.find('div', {'class': 'PZPZlf ssJ7i B5dxMb'})
    form2 = soup.find('div', {'class': 'FLP8od'})
    form3 = soup.find('div', {'class': 'IZ6rdc'})
    forms = [form1, form2, form3]
    possible_names = [form.text for form in forms if form]
    res = ""
    if possible_names and "$" not in possible_names[0]:
        res += f"The initial google suggestion is {possible_names[0]}. This is probably the right answer.\n"
    res += "The webpages that came up that you can click on with the CLICK command are: \n"
    for title in titles:
        res += title + "\n"
    return search_results, res

def clean_string(s):
    return re.sub(r'[^\w\s]', '', s).lower()

async def click(search_results, name):
    titles = [clean_string(await result.inner_text()) for result in search_results]
    cleaned_name = clean_string(name)
    for ii, title in enumerate(titles):
        if title == cleaned_name:
            print("found simple name, clicking")
            await search_results[ii].click()
            return
    print("locating complex name")
    tens = torch.tensor(model.encode([name] + titles)).to(dev)
    tens /= torch.norm(tens, dim=1, keepdim=True)
    sims = tens[1:] @ tens[0]
    best_pos = torch.argmax(sims).item()
    print("found complex name")
    await search_results[best_pos].click()

async def ask_agent(prompt):
    pipe = pipeline("text-generation", model="tanliboy/lambda-qwen2.5-14b-dpo-test", max_new_tokens=1000, device=dev)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless= False, slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.google.com/")

    messages = []

    messages.append({"role":"user","content":init_prompt})
    # print(pipe(messages)[0]["generated_text"][-1]["content"])
    last_search_results = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless= False, slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.google.com/")
        while True:
            w()
            print("done waiting")
            search_prompt = "What is the best search query to achieve this task: {}. Justify your response, and then end with SEARCH and then your query."
            messages = [{"role":"user","content":search_prompt}]
            response = pipe(messages)[0]["generated_text"][-1]["content"]
            if "SEARCH" in response:
                match = re.search("SEARCH(.*)", response)
                payload = match.group(1)
                last_search_results, info = await search(page, payload) 
            messages.append({"role":"assistant","content":response})
            if "SEARCH" in response:
                match = re.search("SEARCH(.*)", response)
                payload = match.group(1)
                last_search_results, info = await search(page, payload)
                next_prompt = coordinator_next_prompt.format(info)
                messages.append({"role":"interface","content":next_prompt})
                print("------------------------------------------")
                print(messages[-1]['content'])
            elif "CLICK" in response:
                print("got a click")
                match = re.search("CLICK(.*)", response)
                payload = match.group(1)
                print("awaiting click")
                await click(last_search_results, payload)
                print("got clicked, now loading content")
                info = BeautifulSoup(await page.content(), 'html.parser').get_text()
                next_prompt = coordinator_next_prompt.format(info)
                messages.append({"role":"interface","content":next_prompt})
                print("------------------------------------------")
                print(messages[-1]['content'])
            # elif "GOBACK" in response:
            #     goback()
            #     next_prompt = coordinator_next_prompt.format("You have successfully gone back a webpage.")
            #     messages.append({"role":"interface","content":next_prompt})
            elif "RES" in response:
                match = re.search("RES(.*)", response)
                # return match.group(1)
                print(match.group(1))
            else:
                next_prompt = coordinator_next_prompt.format("Error, no command given in last response")
                messages.append({"role":"interface","content":next_prompt})
                print("------------------------------------------")
                print(messages[-1]['content'])

asyncio.run(ask_agent("Find me the president of University of Illinois."))


"""
[{'generated_text': [{'role': 'user', 'content': 'Who are you?'}, {'role': 'assistant', 'content': "I am Qwen, a large language model created by Alibaba Cloud. I'm designed to assist with a wide range of tasks including but not limited to generating text, answering questions, providing information, and engaging in conversation on various topics. How can I help you today?"}]}]
"""