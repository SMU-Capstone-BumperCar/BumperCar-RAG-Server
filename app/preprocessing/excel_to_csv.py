import pandas as pd

df = pd.read_excel("app/data/hospital_review_dataset.xlsx", engine="openpyxl")

df.to_csv("app/data/hospital_review_dataset.csv", index=False, encoding='utf-8-sig')
