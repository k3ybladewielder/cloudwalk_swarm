# from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForImageTextToText
import logging
import os
import torch
import yaml
from llama_cpp import Llama
import traceback

# ----------- SETUP -----------
import warnings
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from functions import fn_rebuild_vector_store

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

EMBEDDING_MODEL     = config.get('EMBEDDING_MODEL')
LLM_MODEL           = config.get('LLM_MODEL')
LLM_MODEL_GGUF      = config.get('LLM_MODEL_GGUF')
LLM_MODEL_FILE      = config.get('LLM_MODEL_FILE')

REBUILD_VECTOR_STORE= config.get('REBUILD_VECTOR_STORE', False)
CHUNK_SIZE          = config.get('CHUNK_SIZE', 500)
CHUNK_OVERLAP       = config.get('CHUNK_OVERLAP', 50)
CACHE_FOLDER        = config.get('CACHE_FOLDER', './cache')
URL_LIST            = config.get('URL_LIST', [])
VS_BASE             = config.get('VS_BASE', './vectorstore')

    
def get_llm():
    logger.info(f"Carregando modelo do HuggingFace: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL, 
    cache_dir=CACHE_FOLDER)
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        cache_dir=CACHE_FOLDER,
        device_map="auto",
        torch_dtype=torch.float16
    )

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        #eos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=text_pipeline)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=CACHE_FOLDER)

def load_vector_store():
    logger.info("Loading FAISS vector store...")
    embedding_model = get_embedding_model()
    faiss_file = os.path.join(VS_BASE, "index.faiss")
    pkl_file = os.path.join(VS_BASE, "index.pkl")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Arquivos .faiss e .pkl não encontrados em {VS_BASE}")
    return FAISS.load_local(VS_BASE, embedding_model, allow_dangerous_deserialization=True)

def build_specialist_agents(vectorstore, llm):
    
    template_base = (
    "Você é um especialista da InfinityPay. Use o contexto abaixo para responder à pergunta de forma clara e direta.\n\n"
    "Contexto: {context}\n\nPergunta: {question}\n\nResposta:")
    
    template_base = (
    "<s>"
    "<TASK>\n"
    "Você é um especialista da InfinityPay. Use o contexto abaixo para responder à pergunta de forma clara e direta.\n"
    "</TASK>\n\n"
    
    "<CONTEXT>\n"
    "{context}\n"
    "</CONTEXT>\n"
    
    "<QUESTION>\n"
    "{question}\n"
    "</QUESTION>\n"
    
    "<ANSWER>\n"
    "</s>"
    )



    prompt_template = PromptTemplate(template=template_base, input_variables=["context", "question"])

    def make_agent():
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )

    return {
        "GENERIC": Tool(name="GENERIC", func=make_agent().run, description="Agente genérico sobre a InfinityPay."),
        "MAQUININHA": Tool(name="MAQUININHA", func=make_agent().run, description="Especialista em maquininhas."),
        "COBRANCA_ONLINE": Tool(name="COBRANCA_ONLINE", func=make_agent().run, description="Especialista em cobranças online."),
        "PDV_ECOMMERCE": Tool(name="PDV_ECOMMERCE", func=make_agent().run, description="Especialista em PDV e ecommerce."),
        "CONTA_DIGITAL": Tool(name="CONTA_DIGITAL", func=make_agent().run, description="Especialista em conta digital, Pix, boleto, cartão, etc.")
    }

def load_react_agent(llm):
    if not SERPAPI_API_KEY or SERPAPI_API_KEY == "sua_serpapi_key":
        return None
    try:
        react_tool = Tool(
            name="WebSearch",
            func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run,
            description="Agente para busca na Internet."
        )
        return initialize_agent(
            tools=[react_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )
    except Exception as e:
        logger.error(f"Erro no ReAct: {e}")
        return None

def fallback_fn(input_text: str, llm) -> str:
    #prompt_text = (
    #    "A seguinte pergunta do usuário não pode ser direcionada para um agente específico.\n"
    #    "Responda de forma geral e amigável, informando que a equipe de suporte pode ajudar.\n"
    #    f"\n\nPergunta: {input_text}"
    #)
    
    #prompt_text = (
    #"A seguinte pergunta do usuário não pode ser direcionada para um agente específico.\n"
    #"Responda de forma geral e amigável, informando que a equipe de suporte pode ajudar.\n"
    #f"\n\nPergunta: {input_text}\n\nResposta:\n<s>"
    #)
    
    prompt_text = (
    "<s>"
    "<NOTICE>\n"
    "A seguinte pergunta do usuário não pode ser direcionada para um agente específico.\n"
    "Responda de forma geral e amigável\n"
    "</NOTICE>\n"
    f"<QUESTION>\n{input_text}\n</QUESTION>\n"
    "<ANSWER>\n"
    "</s>"
    )

    try:
        response = llm.invoke(prompt_text)
        clean_response = response.strip().split("<eos>")[0].strip()
        return clean_response.replace("[Assistant]:", "").strip()
    except Exception as e:
        return "Desculpe, não consegui processar sua solicitação agora."

def build_router_chain(llm, tokenizer):
    return None  # Roteador baseado em palavras-chave substitui LLMChain

#def keyword_router(input_text: str) -> str:
#    keywords_map = {
#        "MAQUININHA": ["maquininha", "máquina", "POS", "pagamento físico"],
#        "COBRANCA_ONLINE": ["link de pagamento", "cobrança online", "pagamento online", "checkout"],
#        "PDV_ECOMMERCE": ["PDV", "ecommerce", "venda online", "loja virtual"],
#        "CONTA_DIGITAL": ["conta digital", "pix", "boleto", "transferência", "cartão"]
#    }
#    input_lower = input_text.lower()
#    for agent, keywords in keywords_map.items():
#        if any(keyword.lower() in input_lower for keyword in keywords):
#            return agent
#    return "GENERIC"

def keyword_router(input_text: str) -> str:
    keywords_map = {
        "MAQUININHA": ["maquininha", "máquina", "POS", "pagamento físico"],
        "COBRANCA_ONLINE": ["link de pagamento", "cobrança online", "pagamento online", "checkout"],
        "PDV_ECOMMERCE": ["PDV", "ecommerce", "venda online", "loja virtual"],
        "CONTA_DIGITAL": ["conta digital", "pix", "boleto", "transferência", "cartão"]
    }
    input_lower = input_text.lower()
    for agent, keywords in keywords_map.items():
        if any(keyword in input_lower for keyword in keywords):
            return agent
    return "GENERIC"  # ou "Fallback" se quiser forçar atendimento humano


# def swarm_router(input_text: str, tools: dict, router_chain, llm) -> str:
#     try:
#         agent_name = keyword_router(input_text)
#         selected_tool = tools.get(agent_name, tools["Fallback"])
#         if agent_name == "Fallback":
#             return selected_tool.func(input_text, llm)
#         elif selected_tool.func:
#             return selected_tool.run(input_text)
#         else:
#             return fallback_fn(input_text, llm)
#     except Exception as e:
#         return fallback_fn(input_text, llm)

def swarm_router(input_text: str, tools: dict, router_chain, llm) -> str:
    try:
        agent_name = keyword_router(input_text)
        selected_tool = tools.get(agent_name)

        if selected_tool and selected_tool.func:
            return selected_tool.run(input_text)
        else:
            return fallback_fn(input_text, llm)

    except Exception as e:
        return fallback_fn(input_text, llm)

def main():
    logger.info("Inicializando Swarm...")
    try:
        llm = get_llm()
        tokenizer = llm.pipeline.tokenizer
    except Exception as e:
        logger.error("Erro ao carregar LLM.")
        print(traceback.print_exc())
        return

    try:
        vectorstore = load_vector_store()
    except Exception as e:
        logger.error("Erro ao carregar vectorstore.")
        print(traceback.print_exc())
        vectorstore = None

    specialists = build_specialist_agents(vectorstore, llm) if vectorstore else {}

    if react_agent:
        tools["ReAct"] = Tool(name="ReAct", func=react_agent.run, description="Busca externa na web.")

    router_chain = build_router_chain(llm, tokenizer)

    tools = {}
    tools.update(specialists)
    tools.update(react_agent)
    if react_agent:
        tools["ReAct"] = Tool(name="ReAct", func=react_agent.run, description="Busca externa na web.")
    tools["Fallback"] = Tool(name="Fallback", func=lambda x: fallback_fn(x, llm), description="Fallback generalista.")

    print("\n\nSwarm de agentes iniciado. Digite sua pergunta ou 'sair'.")
    while True:
        query = input("\nUsuário: ")
        if query.strip().lower() == "sair":
            break
        resposta = swarm_router(query, tools, router_chain, llm)
        print(f"\nSwarm: {resposta}")

if __name__ == "__main__":
    main()
