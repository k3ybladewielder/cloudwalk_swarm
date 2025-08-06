# CloudWalk Agent Swarm 🐝

## Description
Swarm of conversational customer service agents. These include:
- **ROUTER**: Receives the customer's message (input) and directs it to the most appropriate agent.
- **GENERIC**: Generic agent for InfinityPay.
- **MAQUININHA**: Specialist in POS terminals.
- **ONLINE_COBRANCA**: Specialist in online billing.
- **PDV_ECOMMERCE**: Specialist in POS and ecommerce.
- **CONTA_DIGITAL**: Specialist in digital accounts, Pix, boleto, cards, etc.
- **Fallback**: Generalist agent, in case the ROUTER fails to direct to other agents.

## How to use:
- Clone this repository into your local development environment.
- Install ``uv`` with pip: ``pip install uv``
- Create a .env file with the environment variables ``HF_TOKEN`` and ``SERPAPI_API_KEY``.
- Create your virtual environment with the ``uv venv .venv``.
- Initialize your virtual environment:
  - Linux/macOS: ``source .venv/bin/activate``
  - Windows (cmd.exe): ``.venv\Scripts\activate``
  - Windows (PowerShell): ``.venv\Scripts\Activate.ps1``
- Install the required libraries with ``uv pip install -r requirements.txt``.
- Application
  - Run the app locally with ``python3 app-gradio.py``;
  - Acess the deployd version on Huggingface Spaces: https://huggingface.co/spaces/k3ybladewielder/cloudwalk_swarm (WIP)

## Deploy:
- tbd

## Params
- LLM: Default model used is ``"Qwen/Qwen3-0.6B"``, to change the model, just change the ```LLM_MODEL``` parameter in the ```config.yaml``` file.
- Vector Store: The parameter ``REBUILD_VECTOR_STORE`` to build the vector stores that is the knowledge base is set ``True`` by default, That is, every time the application is deployed or started locally, the process of creating and storing the vector store will be executed. To learn more, check the ``functions.py`` file.
- Other parameters related to vector store such as sites that serve as source (``URL_LIST``), ``CHUNK_SIZE`` and ``CHUNK_OVERLAP`` can be checked in the ``config.yaml`` file

## Demo
<img src="demo.gif"> 
