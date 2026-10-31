from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever;
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import tempfile

load_dotenv()


embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# Synthetic product documents for testing semantic and exact-keyword retrieval.
documents = [
    Document(
        page_content=(
            "The Atlas router, serial number ATL-RTR-2026-001, supports VPN "
            "(Virtual Private Network) connections and QoS (Quality of Service) "
            "to prioritize voice traffic during network congestion."
        ),
        metadata={"source": "sample_docs/atlas_router_manual.txt"},
    ),
    Document(
        page_content=(
            "The Beacon microphone, serial number BCN-MIC-2026-002, uses DSP "
            "(Digital Signal Processing) and AEC (Acoustic Echo Cancellation) "
            "to reduce echo during voice assistant conversations."
        ),
        metadata={"source": "sample_docs/beacon_microphone_datasheet.txt"},
    ),
    Document(
        page_content=(
            "The Cedar storage appliance, serial number CDR-NAS-2026-003, "
            "provides NAS (Network Attached Storage) with RAID (Redundant Array "
            "of Independent Disks) for storing audio recordings and backups."
        ),
        metadata={"source": "sample_docs/cedar_storage_guide.txt"},
    ),
    Document(
        page_content=(
            "The Delta power unit, serial number DLT-UPS-2026-004, is a UPS "
            "(Uninterruptible Power Supply). Its AVR (Automatic Voltage "
            "Regulation) stabilizes power for servers during voltage fluctuations."
        ),
        metadata={"source": "sample_docs/delta_power_specification.txt"},
    ),
    Document(
        page_content=(
            "The Ember inference workstation, serial number EMB-GPU-2026-005, "
            "uses a GPU (Graphics Processing Unit) to run ASR (Automatic Speech "
            "Recognition) and TTS (Text-to-Speech) models locally."
        ),
        metadata={"source": "sample_docs/ember_workstation_setup.txt"},
    ),
]


def hybrid_search(query):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model,persist_directory=tempfile.mkdtemp())
    semantic_retriever = vectorstore.as_retriever(
        search_type = "similarity", search_kwargs={"k":3},
    )
    bmretriever = BM25Retriever.from_documents(documents=documents)
    ensemble = EnsembleRetriever(retrievers=[semantic_retriever, bmretriever], weights=[0.5,0.5])
    res = ensemble.invoke(query)
    return res


if __name__ == '__main__':
    res = hybrid_search("What do you know about GPUs ?")
    for r in res:
        print(f"{r.page_content}")



