import json

with open("presidents.json","r") as file:
    data = json.load(file)

with open("presidents_verbose.json", "r") as file:
    data2 = json.load(file)

ed = 0
ed2 = 0
es = 0
for d, d2 in zip(data, data2):
    if d["Name"] == "ERROR" and d2["Name"] == "ERROR":
        es += 1
    elif d["Name"] == "ERROR":
        ed += 1
    elif d2["Name"] == "ERROR":
        ed2 += 1

print(f"Length of data is {len(data)}, same error is {es}, just d1 error is {ed}, just d2 error is {ed2}")