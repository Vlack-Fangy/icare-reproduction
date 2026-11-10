import json
import tempfile
import unittest
from pathlib import Path

from anonymized_inputs import digest, load_dialog


class DraftInputTest(unittest.TestCase):
    def fixture(self, root):
        source = 'Hi Alex.\nI cannot sleep.'
        candidate = 'Hi Jo.\nI cannot sleep.'
        doc = {'source_sha256':digest(source), 'utterances':[
            {'speaker':'SPEAKER_00','char_start':0,'char_end':8},
            {'speaker':'SPEAKER_01','char_start':9,'char_end':24}],
            'edits':[{'edit_id':'e1','char_start':3,'char_end':7,'replacement':'Jo'}],
            'analysis':{'transformation':{'status':'complete','draft_revision':'draft1'},
                'drafts':[{'revision_id':'draft1','status':'complete',
                           'text_sha256':digest(candidate),'applied_edit_ids':['e1']}]}}
        (root/'source.txt').write_text(source)
        (root/'output.txt').write_text(candidate)
        (root/'document.json').write_text(json.dumps(doc))
        return doc

    def test_edit_length_changes_preserve_speaker_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            dialog, _ = load_dialog(root)
            self.assertEqual(dialog,'#Therapist#: Hi Jo. #Patient#: I cannot sleep. ')
            self.assertNotIn('Alex',dialog)

    def test_partial_candidate_cannot_enter_full_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); doc=self.fixture(root)
            doc['analysis']['transformation']['status']='partial'
            (root/'document.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'Incomplete anonymization'):
                load_dialog(root)

    def test_changed_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            (root/'output.txt').write_text('some other text')
            with self.assertRaisesRegex(ValueError,'verified revision'):
                load_dialog(root)

    def test_upstream_non_dialogue_omissions_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); doc=self.fixture(root)
            doc['utterances'][1]['speaker']='NARRATOR'
            (root/'document.json').write_text(json.dumps(doc))
            dialog, provenance=load_dialog(root)
            self.assertEqual(dialog,'#Therapist#: Hi Jo. ')
            self.assertEqual(provenance['upstream_formatter_omitted_speaker_turns'],{'NARRATOR':1})


if __name__ == '__main__':
    unittest.main()
