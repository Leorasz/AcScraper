import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

async def get_page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless= False, slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://climas.illinois.edu/directory/profile/tkilleen#:~:text=Biography,with%20national%20scientific%20research%20agencies.")
        print(page)
        _ = input(":")
        info = BeautifulSoup(await page.content(), 'html.parser').get_text()
        print(info)

asyncio.run(get_page())