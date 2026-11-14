from langchain_core.tools import tool
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import tempfile
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState
from typing import Optional


from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import re





load_dotenv()
DEFAULT_HUB_PATH = "~/.local/share/voice-assistant/hub"
HUB_PATH = os.environ.get("VOICE_ASSISTANT_HUB_PATH", DEFAULT_HUB_PATH)

hub_path = Path(HUB_PATH)

@tool
def read_local_file(filename:str) -> str:
    """A tool to search for a filename and return its content.
    args: 
    - filename: the path to the file.
    
    returns: the content of the file
    """
    path = Path(filename)
    if not path.is_file():
        return {"error":"No such file was found"}
    return path.read_text(encoding="utf-8")

@tool
def google_search(query:str) -> dict:
    """Google search tool"""
    results = GoogleSerperAPIWrapper(k=5).results(query)
    return results


@tool
def create_note(content:str, project_id:str | None = None, source_path:str | None = None, tags:list[str] | None  = None) :
    """A tool to create a new user note in the hub.
    args:
    - content: the content of the note.
    - project_id: the project id.
    - source_path: the path to the project.
    - tags: a list of optional tags.
    """
    hub_path = Path(HUB_PATH)
    hub_path.mkdir(parents=True, exist_ok=True)
   
    if project_id is None:
        project_id = str(Path.cwd()).split("/")[-1]
    if source_path is None:
        source_path = str(Path.cwd())

    project_path = hub_path / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    index_path = hub_path / f"index.md"
    if not index_path.exists():
        index_path.write_text("# Projects metadata:\n")
    


    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    note_id = uuid4()
    note = f"""# Example Note 
    ---
    id: {note_id} 
    project_id: {project_id}
    created_at: {timestamp}
    source: {source_path}
    tags: {", ".join([t for t in tags])}
    ---

# Content:
    This note is a demonstration.
    """

    proj_content_header = f"""# Project Metadata:
    project_id: {project_id}
    created_at: {timestamp}
# Related Notes:
"""

    

    path = project_path / f"{note_id}.md"
    with path.open("x", encoding="utf-8") as file:
        file.write(note)
    proj_path_md = project_path / f"{project_id}.md"
    if not proj_path_md.exists():
        with proj_path_md.open("x", encoding="utf-8") as file:
            file.write(proj_content_header)
        with index_path.open("a", encoding="utf-8") as file:
            file.write(f"[{datetime.now().astimezone().isoformat(timespec="seconds")}]: added [[{project_id}]] to {project_path}")
    with proj_path_md.open("a", encoding="utf-8") as file:
        file.write(f"[{datetime.now().astimezone().isoformat(timespec="seconds")}]: added [[{note_id}]] to {project_path}\n")

   
   
@tool
def list_notes(project_id:str | None = None, limit:int =10) -> list:
    """A tool to list notes of the current project.
    args:
    - project_id: the project id.
    - limit: maximum number of notes to return."""
    notes = []
    if project_id is None:
        project_id = str(Path.cwd()).split("/")[-1]
    project_path = hub_path / f"{project_id}"
    if not project_path.exists():
        raise FileExistsError(f"No directory with the name {project_path} was found.")
    for path in project_path.iterdir():
        if path.is_file():
            if path.name == f"{project_id}.md":
                continue
            content = path.read_text()
            match = re.search(r"^[ \t]*created_at:[ \t]*([^\r\n]+)", content, re.MULTILINE)
            if match:
                value = match.group(1).strip().strip("\"'")
            else:
                value = None
            note_content = content.split("# Content")[-1]
            notes.append((note_content, value))
    return sorted(notes, key=lambda val: val[1], reverse=True)[limit-1::-1]

@tool
def search_note(query:str, project_id:str| None = None, limit:int = 10):
    """A tool to search for notes relevant to a given query.
    args:
    - query: user query defining the note content
    - project_id: the project in which the notes exists.
    - limit: maximum result count.
    """
    if project_id is None:
        project_id = str(Path.cwd()).split("/")[-1]
    query_words = set(re.findall(r"\w+", query.casefold()))
    if not query_words:
        return []
    results = []
    for path in (hub_path / f"{project_id}" ).glob("*.md"):
        if path.name == f"{project_id}.md":
            continue

        content = path.read_text(encoding="utf-8")
        note_words = set(re.findall(r"\w+",content.casefold()))
        
        score = len(note_words & query_words)
        if score > 0:
            results.append({
                'id': str(path).split("/")[-1][:-3],
                'path': str(path),
                'score': score,
                'content': content,
            })

    results.sort(key=lambda item: (-item["score"], item["path"]))
    return results[:limit]
        
    

@tool
def update_note(note_id: str, content:str, tags: list[str] = []):
    """A tool to update a written note with new content.
    args:
    - note_id: the id of the note to update.
    - content: the new content.
    - tags: a list of tags.
    """




def build_chain():
    prompt_template = PromptTemplate.from_template("""Search the web for the following topic, and proive your answer.
                                                   topic: {topic}""")
    class WebResultTemplate(BaseModel):
        result: str = Field(..., description="The result of the web page")
        link: Optional[str] = Field(description="the link to the web page")

    model = ChatOllama(model='qwen3.5:4b')
    agent = create_agent(
        model=model,
        tools=[google_search],
        system_prompt="Use search when you don't know the answer.",
    )
    chain = prompt_template | model
    res = agent.invoke({"messages":[{"role":"user","content":"What are good phones below 500 dollars? in 2026"}]})
    print(res['messages'][-1].content)




if __name__ == '__main__':
    load_dotenv()
    #build_chain()
    notes = list_notes.invoke(input={})
    query = 'bla to the weird '
    res = search_note.invoke(input= {
        'query':query,
    })
    print(res)



    
    