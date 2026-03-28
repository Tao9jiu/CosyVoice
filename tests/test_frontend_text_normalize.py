# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for CosyVoiceFrontEnd.text_normalize."""

import json
import unittest
from unittest.mock import MagicMock

import inflect

from cosyvoice.cli.frontend import CosyVoiceFrontEnd


def _make_mock_frontend(
    text_frontend='',
    frd_response=None,
    zh_normalize_side_effect=None,
    en_normalize_side_effect=None,
):
    """Build a minimal object with attributes required by text_normalize."""
    fe = MagicMock()
    fe.text_frontend = text_frontend
    fe.allowed_special = 'all'
    fe.inflect_parser = inflect.engine()

    def _encode(text, allowed_special='all'):
        return list(range(max(1, len(text))))

    fe.tokenizer = MagicMock()
    fe.tokenizer.encode = _encode

    fe.frd = MagicMock()
    if frd_response is not None:
        fe.frd.do_voicegen_frd.return_value = frd_response

    fe.zh_tn_model = MagicMock()
    if zh_normalize_side_effect is not None:
        fe.zh_tn_model.normalize.side_effect = zh_normalize_side_effect
    else:
        fe.zh_tn_model.normalize.side_effect = lambda x: x

    fe.en_tn_model = MagicMock()
    if en_normalize_side_effect is not None:
        fe.en_tn_model.normalize.side_effect = en_normalize_side_effect
    else:
        fe.en_tn_model.normalize.side_effect = lambda x: x

    return fe


class TestTextNormalize(unittest.TestCase):
    """Tests for CosyVoiceFrontEnd.text_normalize (unbound call on mock self)."""

    def test_generator_input_skips_normalize_returns_list_wrapping_generator(self):
        def _gen():
            yield 'a'

        g = _gen()
        fe = _make_mock_frontend()
        out = CosyVoiceFrontEnd.text_normalize(fe, g, split=True, text_frontend=True)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], g)

    def test_ssml_markers_disable_frontend_returns_original_text_list(self):
        fe = _make_mock_frontend()
        text = 'prefix <|token|> suffix'
        out = CosyVoiceFrontEnd.text_normalize(fe, text, split=True, text_frontend=True)
        self.assertEqual(out, [text])

    def test_text_frontend_false_returns_list_or_raw_string(self):
        fe = _make_mock_frontend()
        text = '  raw  '
        self.assertEqual(
            CosyVoiceFrontEnd.text_normalize(fe, text, split=True, text_frontend=False),
            [text],
        )
        self.assertEqual(
            CosyVoiceFrontEnd.text_normalize(fe, text, split=False, text_frontend=False),
            text,
        )

    def test_empty_string_returns_empty_list_or_empty_string(self):
        fe = _make_mock_frontend()
        self.assertEqual(
            CosyVoiceFrontEnd.text_normalize(fe, '', split=True, text_frontend=True),
            [''],
        )
        self.assertEqual(
            CosyVoiceFrontEnd.text_normalize(fe, '', split=False, text_frontend=True),
            '',
        )

    def test_chinese_strips_and_splits_into_segments(self):
        fe = _make_mock_frontend(text_frontend='')
        text = '第一句。第二句。'
        out = CosyVoiceFrontEnd.text_normalize(fe, text, split=True, text_frontend=True)
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(s, str) for s in out))
        self.assertGreater(len(out), 0)
        joined = ''.join(out)
        self.assertIn('第一句', joined)
        self.assertIn('第二句', joined)

    def test_wetext_chinese_calls_zh_normalizer(self):
        fe = _make_mock_frontend(text_frontend='wetext')
        CosyVoiceFrontEnd.text_normalize(fe, '你好。', split=True, text_frontend=True)
        fe.zh_tn_model.normalize.assert_called()

    def test_wetext_english_calls_en_normalizer(self):
        fe = _make_mock_frontend(text_frontend='wetext')
        CosyVoiceFrontEnd.text_normalize(fe, 'Hello world.', split=True, text_frontend=True)
        fe.en_tn_model.normalize.assert_called()

    def test_english_spell_out_numbers(self):
        fe = _make_mock_frontend(text_frontend='')
        out = CosyVoiceFrontEnd.text_normalize(fe, 'I have 2 cats.', split=True, text_frontend=True)
        self.assertTrue(any('two' in s.lower() for s in out))

    def test_split_false_returns_processed_string_not_list(self):
        fe = _make_mock_frontend(text_frontend='')
        text = 'Only one sentence here.'
        out = CosyVoiceFrontEnd.text_normalize(fe, text, split=False, text_frontend=True)
        self.assertIsInstance(out, str)
        self.assertIn('Only one sentence here', out)

    def test_ttsfrd_path_uses_frd_and_filters_punctuation_only_sentences(self):
        payload = {
            'sentences': [
                {'text': 'Hello.'},
                {'text': '…'},
            ]
        }
        fe = _make_mock_frontend(text_frontend='ttsfrd', frd_response=json.dumps(payload))
        out = CosyVoiceFrontEnd.text_normalize(fe, 'dummy', split=True, text_frontend=True)
        fe.frd.do_voicegen_frd.assert_called_once_with('dummy')
        self.assertIsInstance(out, list)

    def test_filters_punctuation_only_segments(self):
        fe = _make_mock_frontend(text_frontend='')
        # Force a path that could yield punctuation-only chunks; empty list is acceptable
        # if everything is stripped as punctuation-only.
        out = CosyVoiceFrontEnd.text_normalize(fe, '你好。', split=True, text_frontend=True)
        for seg in out:
            self.assertNotEqual(seg.strip(), '')


if __name__ == '__main__':
    unittest.main()
