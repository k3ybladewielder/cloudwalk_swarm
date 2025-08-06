from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import get_llm, load_vector_store, build_specialist_agents, load_react_agent, fallback_fn, swarm_router
from langchain.agents import Tool
import logging

app = FastAPI(title="Swarm de Agentes - InfinityPay")

# Inicializa logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Inicialização dos componentes ----------
try:
    llm = get_llm()
    vectorstore = load_vector_store()
    specialists = build_specialist_agents(vectorstore, llm)
    tools = {}

    # Adiciona agente de busca na web, se disponível
    react_agent = load_react_agent(llm)
    if react_agent:
        tools["WEB_SEARCHER"] = Tool(name="WEB_SEARCHER", func=react_agent.run, description="Busca externa na web.")
    
    tools.update(specialists)
    tools["Fallback"] = Tool(name="Fallback", func=lambda x: fallback_fn(x, llm), description="Fallback generalista.")
    
    logger.info(f"Agentes carregados: {list(tools.keys())}")

except Exception as e:
    logger.error(f"Erro durante inicialização da API: {e}")
    raise

# ---------- Modelo da requisição ----------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    agent: str
    answer: str

# ---------- Rota principal ----------
@app.post("/ask", response_model=QueryResponse)
def ask_question(query: QueryRequest):
    try:
        agent_name = "Fallback"
        response = swarm_router(query.question, tools, llm)
        # Tentativa de detectar o agente usado
        for name, tool in tools.items():
            if tool.run == tools.get(name).run:
                agent_name = name
        return QueryResponse(agent=agent_name, answer=response)

    except Exception as e:
        logger.error(f"Erro na pergunta: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar sua pergunta.")

# Rota básica para checagem de status
@app.get("/")
def root():
    return {"message": "API da Swarm de Agentes InfinityPay está ativa"}

