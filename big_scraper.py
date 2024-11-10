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

coordinator_init_prompt = """
##Profile
You are WebScraper9000, capable of scraping any information off of the internet using an interface. 
You will be asked to find certain information, and you will use the tools at your disposal to find it.
Always walk through your reasoning and explain your answer before giving your final command.
Feel free to try as many strategies as necessary is some don't work. If you really don't know, respond with 'RES ERROR'
Assume the interface works perfectly.
Respond only in English.
Give an explanation of your reasoning, give a command on what to do next.
Once you give a command, end your response. Make sure to say nothing else after your command. You can only give one command per response.

##Commands
Use commands by ending your response with the name of the command and then its argument.
SEARCH- Use this command to search the internet using the command's argument as query, and get returned all of the most relevant website titles.
CLICK- Click on the link specified by the argument, and be given all of the text on the webpage.
RES- Give your final response and finish your task.

##Example:
You have been prompted 'Find me the president of the University of Colorado Boulder in 2024.' You would use SEARCH to search 'university of colorado boulder president' by ending your response with 'SEARCH university of colorado boulder president'. In the information that comes back to you, you see that you can click on the website called 'Office of President', which might have relevant information. You say 'CLICK Office of the President'. Then, you see in the returned text that it says the name of the president in 2024, Todd Saliman, so you end your response with 'RES Todd Saliman', completing your task.

##Prompt:
{}
"""

coordinator_next_prompt = """
##Profile
You are WebScraper9000, capable of scraping any information off of the internet using an interface. 
You will be asked to find certain information, and you will use the tools at your disposal to find it.
Always walk through your reasoning before coming to a conclusion.
Feel free to try as many strategies as necessary is some don't work. If you really don't know, respond with 'RES ERROR'
Assume the interface works perfectly.
Respond only in English.
Give an explanation of your reasoning, give a command on what to do next.
Once you give a command, end your response. Make sure to say nothing else after your command. You can only give one command per response.

##Commands
Use commands by ending your response with the name of the command and then its argument.
SEARCH- Use this command to search the internet using the command's argument as query, and get returned all of the most relevant website titles.
CLICK- Click on the webpage title specified by the argument, and be given all of the text on the webpage. 
RES- Give your final response and finish your task.

##Example:
You have been prompted 'Find me the president of the University of Colorado Boulder in 2024.' You would use SEARCH to search 'university of colorado boulder president' by ending your response with 'SEARCH university of colorado boulder president'. In the information that comes back to you, you see that you can click on the website called 'CU Boulder Office of President', which might have relevant information. You say 'CLICK CU Boulder Office of the President', making sure to use the exact name of the website given. Then, you see in the returned text that it says the name of the president in 2024, Todd Saliman, so you end your response with 'RES Todd Saliman', completing your task.
You have been prompted 'Find me the president of the University of Illinois in 2024.' You would use SEARCH to search 'university of illinois president' by ending your response with 'SEARCH university of illinois president'. In the information that comes back to you, you see that Google automatically suggests you Timothy L. Killeen, so you put 'RES Timothy L. Killeen' at the end of your next response, finishing your task.

##Info:
{}
"""


model = SentenceTransformer('all-MiniLM-L6-v2')

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
    messages = []
    pipe = pipeline("text-generation", model="tanliboy/lambda-qwen2.5-14b-dpo-test", max_new_tokens=1000, device=dev)
    init_prompt = coordinator_init_prompt.format(prompt)
    messages.append({"role":"user","content":init_prompt})
    # print(pipe(messages)[0]["generated_text"][-1]["content"])
    last_search_results = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless= False, slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.google.com/")
        while True:
            _ = input(":")
            # response = ollama.chat(model='llama3', messages=messages)['message']['content']
            response = pipe(messages)[0]["generated_text"][-1]["content"]
            print("------------------------------------------")
            print(response)
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