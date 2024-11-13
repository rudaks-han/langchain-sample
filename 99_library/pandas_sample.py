import pandas as pd
import requests

filename = "파일명.xlsx"
df = pd.read_excel(filename, usecols="A,D,E", sheet_name="faq")

url = "http://localhost:8181/knowledges/kodma_knwl/data/qna"

for index, row in df.iterrows():
    content = ""
    for col_name, cell_value in row.items():
        if col_name == "대분류":
            content += f"[{cell_value}]"
        elif col_name == "question":
            content += f" {cell_value}"
        elif col_name == "answer":
            content += f"\n{cell_value}"
        # print(f"행 {index}, 열 '{col_name}': {cell_value}")
        # pass
    print("_________________")
    print(content)
    data = [
        {
            "accessType": "public",
            "content": content,
            "metadata": {},
            "source": "",
            "userId": "gdhong",
        }
    ]

    response = requests.post(url, json=data)
    print(response)
