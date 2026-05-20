import os
from dotenv import load_dotenv
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if ChatOpenAI is None:
    print(["AVISO] 'langchain-openai' não está instalado."])
elif not api_key:
    print(["AVISO] 'OPENAI_API_KEY' não encontrado no arquivo .env."])
else:
    llm = ChatOpenAI(model=model, temperature=0.2, max_tokens=64)
    resp = llm.invoke("Diga 'Ambiente OK' em uma frase curta")
    print("[LLM]", resp.content)