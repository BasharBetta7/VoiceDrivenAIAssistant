import logging
import warnings
from langchain_core._api import LangChainBetaWarning

logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=LangChainBetaWarning)