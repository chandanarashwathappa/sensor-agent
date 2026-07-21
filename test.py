from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

llm = ChatMistralAI(model="mistral-small-latest")

response = llm.invoke("Explain what predictive maintenance is in one sentence.")

print(response.content)