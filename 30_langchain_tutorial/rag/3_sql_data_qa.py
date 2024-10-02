import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

df = pd.read_csv("titanic.csv")
print(df.shape)
print(df.columns.tolist())

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

engine = create_engine("sqlite:///titanic4.db")
df.to_sql("titanic", engine, index=False)

db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM titanic WHERE Age < 2;"))

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

output = agent_executor.invoke({"input": "what's the average age of survivors"})
print(output)
