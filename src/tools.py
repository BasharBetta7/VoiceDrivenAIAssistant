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
import yaml
from urllib.parse import quote





load_dotenv()
DEFAULT_HUB_PATH = "~/.local/share/voice-assistant/hub"
HUB_PATH = os.environ.get("VOICE_ASSISTANT_HUB_PATH", DEFAULT_HUB_PATH)

hub_path = Path(HUB_PATH).expanduser()

def _project_path(project_id: str) -> Path:
    if not project_id or project_id in {".", ".."} or "/" in project_id or "\\" in project_id:
        raise ValueError("project_id must be a single directory name.")
    path = (hub_path / project_id).resolve()
    if not path.is_relative_to(hub_path.resolve()):
        raise ValueError("Project directory must remain inside the hub.")
    return path


def _note_title(content: str) -> str:
    """Use the first nonempty content line when no title was supplied."""
    return next((line.strip().lstrip("#").strip()[:100]
                 for line in content.splitlines() if line.strip()), "Untitled note")


def _read_note(content: str, fallback_id: str) -> dict:
    # New notes use YAML front matter. Older notes used indented metadata.
    metadata = {}
    body = content
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", content, re.DOTALL)
    if match:
        parsed = yaml.safe_load(match.group(1))
        metadata = parsed if isinstance(parsed, dict) else {}
        body = content[match.end():].strip()
    else:
        header, separator, body = content.partition("# Content:")
        if not separator:
            body = content
        for key in ("id", "title", "created_at"):
            field = re.search(rf"^[ \t]*{key}:[ \t]*([^\r\n]+)", header, re.MULTILINE)
            if field:
                metadata[key] = field.group(1).strip().strip("\"'")
    created_at = metadata.get("created_at")
    return {"id": str(metadata.get("id") or fallback_id),
            "title": str(metadata.get("title") or _note_title(body)),
            "content": body.strip(),
            "created_at": str(created_at) if created_at is not None else None,
            "updated_at": str(metadata["updated_at"]) if metadata.get("updated_at") else None}


def _note_link(title: str, relative_path: str) -> str:
    label = title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{label}]({quote(relative_path, safe='/')})"


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
def create_note(content:str, project_id:str | None = None, source_path:str | None = None, tags:list[str] | None  = None, title:str | None = None) -> dict:
    """A tool to create a new user note in the hub.
    args:
    - content: the content of the note.
    - project_id: the project id.
    - source_path: the path to the project.
    - tags: a list of optional tags.
    - title: a short descriptive title; defaults to the first nonempty content line.
    """
    hub_path.mkdir(parents=True, exist_ok=True)
   
    if project_id is None:
        project_id = str(Path.cwd()).split("/")[-1]
    if source_path is None:
        source_path = str(Path.cwd())

    project_path = _project_path(project_id)
    project_path.mkdir(parents=True, exist_ok=True)
    index_path = hub_path / f"index.md"
    if not index_path.exists():
        index_path.write_text("# Projects metadata:\n")
    


    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    note_id = uuid4()
    title = " ".join(title.split()) if title and title.strip() else _note_title(content)
    metadata = {"id": str(note_id), "title": title, "project_id": project_id,
                "created_at": timestamp, "source": source_path, "tags": tags or []}
    note = "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n\n" + content + "\n"

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
        file.write(f"\n- {timestamp} — {_note_link(title, f'{project_id}/{note_id}.md')}\n")
    with proj_path_md.open("a", encoding="utf-8") as file:
        file.write(f"- {timestamp} — {_note_link(title, f'{note_id}.md')}\n")

    return {"id": str(note_id), "title": title, "path": str(path), "created_at": timestamp}



@tool
def list_notes(project_id:str | None = None, limit:int =10) -> list:
    """A tool to list notes of the current project.
    args:
    - project_id: the project id.
    - limit: maximum number of notes to return."""
    notes = []
    if project_id is None:
        project_id = str(Path.cwd()).split("/")[-1]
    project_path = _project_path(project_id)
    if not project_path.exists():
        return []
    if limit <= 0:
        return []
    for path in project_path.glob("*.md"):
        if path.is_file():
            if path.name == f"{project_id}.md":
                continue
            content = path.read_text(encoding="utf-8")
            notes.append({**_read_note(content, path.stem), "path": str(path)})
    return sorted(notes, key=lambda note: datetime.fromisoformat(note["created_at"]).timestamp()
                  if note["created_at"] else float("-inf"), reverse=True)[:limit]

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
    if not query_words or limit <= 0:
        return []
    results = []
    for path in _project_path(project_id).glob("*.md"):
        if path.name == f"{project_id}.md":
            continue

        content = path.read_text(encoding="utf-8")
        note_words = set(re.findall(r"\w+",content.casefold()))
        
        score = len(note_words & query_words)
        if score > 0:
            results.append({
                'id': _read_note(content, path.stem)['id'],
                'title': _read_note(content, path.stem)['title'],
                'path': str(path),
                'score': score,
                'content': content,
            })

    results.sort(key=lambda item: (-item["score"], item["path"]))
    return results[:limit]
        
    

@tool
def update_note(note_id: str, content: str, tags: list[str] | None = None) -> dict:
    """Replace a note's complete body by its exact ID, searching all hub projects.

    Obtain the ID from list_notes or search_note. Preserve ID, title, creation
    time and source. Omit tags to preserve them; pass [] to clear them.
    Returns updated note metadata, or an error if the ID is missing or ambiguous.
    """
    if not note_id.strip():
        return {"error": "note_id must not be empty."}
    matches = []
    root = hub_path.resolve()
    for path in hub_path.glob("*/*.md"):
        if path.name == f"{path.parent.name}.md" or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            continue
        original = path.read_text(encoding="utf-8")
        parsed = _read_note(original, path.stem)
        if parsed["id"] == note_id:
            matches.append((path, original, parsed))
    if not matches:
        return {"error": f"No note found with ID {note_id!r}."}
    if len(matches) != 1:
        return {"error": f"Multiple notes have ID {note_id!r}; resolve the duplicate ID before updating."}

    path, original, parsed = matches[0]
    front = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", original, re.DOTALL)
    if front:
        metadata = yaml.safe_load(front.group(1)) or {}
    else:
        # Preserve legacy metadata while migrating this note to YAML front matter.
        metadata = {}
        header = original.partition("# Content:")[0]
        for key, value in re.findall(r"^[ \t]*([\w]+):[ \t]*([^\r\n]+)", header, re.MULTILINE):
            metadata[key] = value.strip().strip("\"'")
    metadata.setdefault("id", parsed["id"])
    metadata.setdefault("title", parsed["title"])
    if tags is not None:
        metadata["tags"] = tags
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    updated = "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n\n" + content + "\n"

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(updated)
            file.flush()
            os.fsync(file.fileno())
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"id": parsed["id"], "title": metadata["title"], "path": str(path),
            "updated_at": metadata["updated_at"]}



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



    
    