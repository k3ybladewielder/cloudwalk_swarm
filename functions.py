#----------- SETUP -----------
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import logging
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import yaml
import logging

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------- PARAMS -----------
with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
EMBEDDING_MODEL = config.get('EMBEDDING_MODEL')
LLM_MODEL = config.get('LLM_MODEL')
REBUILD_VECTOR_STORE = config.get('REBUILD_VECTOR_STORE')
CHUNK_SIZE = config.get('CHUNK_SIZE')
CHUNK_OVERLAP = config.get('CHUNK_OVERLAP')
CACHE_FOLDER = config.get('CACHE_FOLDER')
URL_LIST = config.get('URL_LIST')
VS_BASE = config.get('VS_BASE')

# ----------- VECTOR STORE CREATION -----------
def fn_rebuild_vector_store(REBUILD_VECTOR_STORE, URL_LIST, VS_BASE, EMBEDDING_MODEL, CACHE_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP):
    if REBUILD_VECTOR_STORE:
        logger.info("REBUILD_VECTOR_STORE was set True. Recreating the vector store...")
        loader = WebBaseLoader(web_paths=URL_LIST)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=CACHE_FOLDER)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        os.makedirs(VS_BASE, exist_ok=True)
        vector_store.save_local(VS_BASE)
        print()
        logger.info(f"Vector Store saved in the path: {VS_BASE}")
    else:
        logger.info(f"REBUILD_VECTOR_STORE was set False. Using the current vector store...")
    return print(f"End of vector store process")
