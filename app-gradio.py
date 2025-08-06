# app-gradio.py

import logging
import gradio as gr
from dotenv import load_dotenv

# Carregamento das variáveis de ambiente
load_dotenv()

# Importações principais do módulo de agentes
from agents import (
    swarm_router,
    get_llm,
    load_vector_store,
    build_specialist_agents,
    load_react_agent,
    build_router_chain,
    fallback_fn,
    Tool
)

# Configuração de logs
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Variáveis globais
llm = None
tokenizer = None
tools = None
router_chain = None

# Função de setup
def setup():
    global llm, tokenizer, tools, router_chain

    logger.info("Inicializando Swarm via Gradio...")

    try:
        llm = get_llm()
        tokenizer = llm.pipeline.tokenizer
    except Exception as e:
        logger.error("Erro ao carregar LLM.", exc_info=True)
        return "Erro ao carregar o modelo."

    try:
        vectorstore = load_vector_store()
    except Exception as e:
        logger.error("Erro ao carregar vectorstore.", exc_info=True)
        vectorstore = None

    # Monta os agentes especialistas
    specialists = build_specialist_agents(vectorstore, llm) if vectorstore else {}

    # Agente de busca externa (opcional)
    react_agent = load_react_agent(llm)

    # (Opcional) Chain de roteamento
    router_chain = build_router_chain(llm, tokenizer)

    # Monta o dicionário final de ferramentas
    tools_local = {}
    tools_local.update(specialists)
    if react_agent:
        tools_local["ReAct"] = Tool(name="ReAct", func=react_agent.run, description="Busca externa na web.")
        
    tools_local["Fallback"] = Tool(name="Fallback", func=lambda x: fallback_fn(x, llm), description="Fallback generalista.")
    tools = tools_local

# Função usada pelo Gradio para processar mensagens
def gradio_response(user_input, history):
    if not tools:
        return "Agentes ainda não estão prontos. Aguarde o carregamento."
    return swarm_router(user_input, tools, router_chain, llm)

# Inicializa
setup()

# Interface Gradio
if __name__ == "__main__":
    print("Iniciando a interface Gradio...")
    demo = gr.ChatInterface(
        fn=gradio_response,
        type="messages",
        title="Assistente InfinityPay",
        description="Digite uma pergunta relacionada à InfinityPay e receba uma resposta especializada.",
        submit_btn="Enviar Pergunta",
        examples=[
            ["Quais são as soluções da InfinitePay para o meu negócio?"],
            ["Como começar a vender com a InfinitePay?"],
            ["Pessoa Física pode vender com a InfinitePay?"],
            ["Como faço o meu cadastro na InfinitePay?"],
            ["Qual é o prazo de entrega da Maquininha Smart?"],
            ["Quais são as taxas da InfinitePay para CNPJ?"],
            ["Quais são as taxas da InfinitePay para CPF?"],
            ["Quais são as taxas da InfinitePay?"],
            ["Quais bandeiras são aceitas para adquirir as soluções da InfinitePay?"],
            ["Como posso comprar uma Maquininha Smart?"],
            ["Quais modelos de máquinas de cartão posso comprar?"],
            ["Posso ter mais de uma máquina no mesmo CNPJ?"],
            ["Em quanto tempo é feita a análise do meu cadastro?"],
            ["Pago aluguel para usar InfinitePay?"],
        ],
        chatbot=gr.Chatbot(type="messages")
    )
    demo.launch()
