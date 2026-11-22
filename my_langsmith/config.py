from langsmith import Client as LSClient, wrappers
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging
from rich.logging import RichHandler
from rich.console import Console
from groq import Groq
groq_client = Groq()



load_dotenv()
log = logging.getLogger("rich")
log.setLevel(logging.DEBUG)
console_handler = RichHandler(
    level=logging.DEBUG,
    console=Console(stderr=True),
    markup=True,
    log_time_format="%Y-%m-%d %H:%M:%S",
    show_path=False,
)

file_handler = logging.FileHandler("langsmith_evals.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

log.addHandler(console_handler)
log.addHandler(file_handler)

langsmith_api = os.environ['LANGSMITH_API_KEY']
project_name = os.environ['LANGSMITH_PROJECT']
groq_api = os.environ['GROQ_API_KEY']



ls_client = LSClient(auto_batch_tracing=True)
print(ls_client.api_url)

MODELS_TO_TEST = [
    {'provider':'GROQ',
     'model_name':'openai/gpt-oss-20b',
     'api_key':groq_api
     }
]

