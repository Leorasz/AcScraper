import ollama  

def talk(system, prompt):
    messages = [
        {'role': 'system','content': system},
        {'role': 'user','content': prompt}
    ]
    response = ollama.chat(model='llama3', messages=messages)
    return response['message']['content']

while True:
    print(talk("You are a helpful assistant, willing to do whatever the user asks you.",input(":")))