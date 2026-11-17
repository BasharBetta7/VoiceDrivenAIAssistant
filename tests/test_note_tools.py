"""Note tools operate on a temporary hub, without model or network calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import tools
from src.llm_model import LLMChat


class NoteToolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.hub = Path(temporary.name)
        patcher = patch.object(tools, 'hub_path', self.hub)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_create_list_search_round_trip(self):
        saved = tools.create_note.invoke({'content': 'Investigate speaker rejection', 'project_id': 'demo'})
        self.assertIn('Investigate speaker rejection', Path(saved['path']).read_text())
        listed = tools.list_notes.invoke({'project_id': 'demo'})
        self.assertEqual(listed[0]['id'], saved['id'])
        self.assertEqual(listed[0]['content'], 'Investigate speaker rejection')
        found = tools.search_note.invoke({'query': 'speaker', 'project_id': 'demo'})
        self.assertEqual(found[0]['id'], saved['id'])
        json.dumps([saved, listed, found])

    def test_titles_metadata_links_and_duplicate_titles(self):
        from uuid import UUID
        import yaml
        title = 'Speaker [verification]: "short commands"'
        saved = tools.create_note.invoke({'content': 'Original body', 'title': title,
                                          'project_id': 'demo'})
        duplicate = tools.create_note.invoke({'content': 'Another body', 'title': title,
                                              'project_id': 'demo'})
        self.assertNotEqual(saved['id'], duplicate['id'])
        self.assertEqual(UUID(saved['id']).version, 4)
        path = Path(saved['path'])
        self.assertEqual(path.stem, saved['id'])
        metadata = yaml.safe_load(path.read_text().split('---', 2)[1])
        self.assertEqual(metadata['title'], title)
        listed = tools.list_notes.invoke({'project_id': 'demo'})
        self.assertTrue(all(note['title'] == title for note in listed))
        found = tools.search_note.invoke({'query': 'verification', 'project_id': 'demo'})
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]['title'], title)
        self.assertIn(f'(demo/{saved["id"]}.md)', (self.hub / 'index.md').read_text())
        project_index = (self.hub / 'demo' / 'demo.md').read_text()
        self.assertIn(f'({saved["id"]}.md)', project_index)
        self.assertIn('Speaker', project_index)

    def test_legacy_notes_get_title_without_rewriting(self):
        project = self.hub / 'demo'
        project.mkdir()
        path = project / 'legacy.md'
        original = '# Note\n    id: old-id\n    created_at: 2026-11-17T12:00:00+03:00\n# Content:\n    Useful observation'
        path.write_text(original)
        listed = tools.list_notes.invoke({'project_id': 'demo'})
        self.assertEqual(listed[0]['title'], 'Useful observation')
        self.assertEqual(listed[0]['id'], 'old-id')
        self.assertEqual(path.read_text(), original)

    def test_list_orders_newest_first_and_limits(self):
        project = self.hub / 'demo'
        project.mkdir()
        for day in (1, 3, 2):
            (project / f'{day}.md').write_text(
                f'created_at: 2026-11-{day:02}T12:00:00+03:00\n# Content:\nnote {day}')
        notes = tools.list_notes.invoke({'project_id': 'demo', 'limit': 2})
        self.assertEqual([note['id'] for note in notes], ['3', '2'])
        self.assertEqual(tools.list_notes.invoke({'project_id': 'missing'}), [])
        self.assertEqual(tools.list_notes.invoke({'project_id': 'demo', 'limit': 0}), [])

    def test_project_cannot_escape_hub(self):
        with self.assertRaises(ValueError):
            tools.create_note.invoke({'content': 'test', 'project_id': '../outside'})

    def test_llm_registers_note_tools(self):
        with patch('src.llm_model.ChatOllama'):
            agent = LLMChat()
        registered = {tool.name: tool for tool in agent.tools}
        for tool in (tools.create_note, tools.list_notes, tools.search_note, tools.update_note):
            self.assertIs(registered[tool.name], tool)
            self.assertIn(tool.name, agent.sys_prompt)

    def test_update_preserves_identity_metadata_and_index_links(self):
        import yaml
        saved = tools.create_note.invoke({'content': 'Old observation', 'title': 'Stable title',
                                          'project_id': 'another-project', 'tags': ['experiment']})
        path = Path(saved['path'])
        before = yaml.safe_load(path.read_text().split('---', 2)[1])
        index = (self.hub / 'index.md').read_text()
        result = tools.update_note.invoke({'note_id': saved['id'], 'content': 'New observation'})
        after = yaml.safe_load(path.read_text().split('---', 2)[1])
        self.assertEqual(result['id'], saved['id'])
        self.assertEqual(result['path'], saved['path'])
        for key, value in before.items():
            self.assertEqual(after[key], value)
        self.assertIn('updated_at', after)
        listed = tools.list_notes.invoke({'project_id': 'another-project'})
        self.assertEqual(listed[0]['content'], 'New observation')
        self.assertEqual(listed[0]['updated_at'], result['updated_at'])
        self.assertEqual((self.hub / 'index.md').read_text(), index)
        tools.update_note.invoke({'note_id': saved['id'], 'content': '', 'tags': []})
        self.assertEqual(yaml.safe_load(path.read_text().split('---', 2)[1])['tags'], [])

    def test_missing_and_duplicate_ids_do_not_write(self):
        saved = tools.create_note.invoke({'content': 'Original', 'project_id': 'demo'})
        path = Path(saved['path'])
        original = path.read_text()
        self.assertIn('error', tools.update_note.invoke({'note_id': '../missing', 'content': 'New'}))
        (path.parent / 'duplicate.md').write_text(original)
        self.assertIn('error', tools.update_note.invoke({'note_id': saved['id'], 'content': 'New'}))
        self.assertEqual(path.read_text(), original)

    def test_update_legacy_note_preserves_metadata(self):
        import yaml
        project = self.hub / 'demo'
        project.mkdir()
        path = project / 'legacy.md'
        path.write_text('# Note\n    id: legacy-id\n    created_at: 2026-11-17T12:00:00+03:00\n    source: /project\n    tags: one, two\n# Content:\nOld text')
        result = tools.update_note.invoke({'note_id': 'legacy-id', 'content': 'Revised text'})
        self.assertEqual(result['title'], 'Old text')
        metadata = yaml.safe_load(path.read_text().split('---', 2)[1])
        self.assertEqual(metadata['source'], '/project')
        self.assertEqual(metadata['tags'], 'one, two')
        self.assertEqual(tools.list_notes.invoke({'project_id': 'demo'})[0]['content'], 'Revised text')

    def test_failed_replace_preserves_original(self):
        saved = tools.create_note.invoke({'content': 'Original', 'project_id': 'demo'})
        path = Path(saved['path'])
        original = path.read_bytes()
        with patch.object(tools.os, 'replace', side_effect=OSError('write failure')):
            with self.assertRaises(OSError):
                tools.update_note.invoke({'note_id': saved['id'], 'content': 'New'})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(path.parent.glob('*.tmp')), [])
