"""Ensure publication transport preserves exact CSV bytes and rejects corruption."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import capture_verification
from export_predictions import export


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.old_here = capture_verification.HERE
        capture_verification.HERE = self.folder
        frame = pd.DataFrame(dict(sha256=['0'*64, 'f'*64], patient=['normal1', 'person2'], y_true=[0, 1],
                                  probability=[.1, .9], prediction=[0, 1], threshold=[.5, .5]))
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.meta = export(frame, self.folder, 'public_test_predictions.csv')
        self.messages = captured.getvalue().splitlines()
        self.messages.append('NOVEMBER_CPU_VERIFICATION ' + json.dumps(dict(
            status='passed', public_prediction_exports={'test': self.meta})))

    def tearDown(self):
        capture_verification.HERE = self.old_here
        self.temporary.cleanup()

    def capture(self, messages):
        log = self.folder / 'synthetic.log'
        log.write_text('\n'.join(f'{i}.0s\t{i}\t{line}' for i, line in enumerate(messages)), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            return capture_verification.capture(log, 'primary')

    def test_exact_reconstruction(self):
        self.assertTrue(self.capture(self.messages)['all_export_byte_hashes_verified'])
        self.assertEqual((self.folder/'public_test_predictions.csv').read_bytes(),
                         (self.folder/'predictions/public_test_predictions.csv').read_bytes())

    def test_modified_score_is_rejected(self):
        messages = list(self.messages)
        block = json.loads(messages[1].removeprefix('PUBLIC_PREDICTION_BLOCK '))
        block['rows'][0]['probability'] = .2
        messages[1] = 'PUBLIC_PREDICTION_BLOCK ' + json.dumps(block)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.capture(messages)
        self.assertFalse((self.folder/'predictions').exists())

    def test_missing_block_is_rejected(self):
        with self.assertRaises(ValueError): self.capture([self.messages[0], self.messages[-1]])


if __name__ == '__main__': unittest.main()
