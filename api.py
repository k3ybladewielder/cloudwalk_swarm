from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from agents import (
    get_llm,
    load_vector_store,
    build_specialist_agents,
    load_react_agent,
    fallback_fn,
    keyword_router,
)
from langchain.agents import Tool
import logging

app = FastAPI(title="InfinityPay Agent Swarm API")

# Log
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Inicialização dos componentes ----------
llm = get_llm()
vectorstore = load_vector_store()
specialists = build_specialist_agents(vectorstore, llm)

tools = {}

react_agent = load_react_agent(llm)
if react_agent:
    tools["WEB_SEARCHER"] = Tool(name="WEB_SEARCHER", func=react_agent.run, description="Busca na web")

tools.update(specialists)

# Fallback
tools["Fallback"] = Tool(name="Fallback", func=lambda x: fallback_fn(x, llm), description="Fallback generalista")

logger.info(f"Agentes disponíveis: {list(tools.keys())}")

# ---------- MODELO DE ENTRADA ----------
class QueryRequest(BaseModel):
    message: str
    user_id: str

# ---------- MODELO DE SAÍDA ----------
class AgentWorkflowStep(BaseModel):
    agent_name: str
    tool_calls: Dict[str, str]

class QueryResponse(BaseModel):
    response: str
    source_agent_response: str
    agent_workflow: List[AgentWorkflowStep]

# ---------- FUNÇÃO: LAYER DE PERSONALIDADE ----------
def apply_personality(raw_response: str, user_id: str) -> str:
    # Aqui você pode aplicar uma camada de estilo/persona
    # Exemplo básico: resposta mais amigável
    return f"Claro! 😊 {raw_response}"

# ---------- ENDPOINT ----------
@app.post("/ask", response_model=QueryResponse)
def ask_agent_swarm(payload: QueryRequest):
    try:
        message = payload.message
        user_id = payload.user_id

        # Seleciona o agente com base em palavras-chave
        agent_name = keyword_router(message)
        selected_tool = tools.get(agent_name)

        if selected_tool and selected_tool.func:
            raw_response = selected_tool.run(message)
        else:
            agent_name = "Fallback"
            raw_response = fallback_fn(message, llm)

        # Aplicar camada de personalidade (ajuste do estilo de resposta)
        personality_response = apply_personality(raw_response, user_id)

        # Montar histórico de execução
        workflow_log = [
            AgentWorkflowStep(
                agent_name=agent_name,
                tool_calls={agent_name: raw_response}
            )
        ]

        return QueryResponse(
            response=personality_response,
            source_agent_response=raw_response,
            agent_workflow=workflow_log
        )

    except Exception as e:
        logger.error(f"Erro ao processar a pergunta: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar sua pergunta.")
