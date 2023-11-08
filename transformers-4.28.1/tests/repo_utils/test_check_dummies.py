# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import unittest

git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_dummies  # noqa: E402
from check_dummies import (create_dummy_files,  # noqa: E402
                           create_dummy_object, find_backend, read_init)

# Align TRANSFORMERS_PATH in check_dummies with the current path
check_dummies.PATH_TO_TRANSFORMERS = os.path.join(git_repo_path, "src", "transformers")

DUMMY_CONSTANT = """
{0} = None
"""

DUMMY_CLASS = """
class {0}(metaclass=DummyObject):
    _backends = {1}

    def __init__(self, *args, **kwargs):
        requires_backends(self, {1})
"""


DUMMY_FUNCTION = """
def {0}(*args, **kwargs):
    requires_backends({0}, {1})
"""


class CheckDummiesTester(unittest.TestCase):
    def test_find_backend(self):
        no_backend = find_backend('    _import_structure["models.albert"].append("AlbertTokenizerFast")')
        self.assertIsNone(no_backend)

        simple_backend = find_backend("    if not is_tokenizers_available():")
        self.assertEqual(simple_backend, "tokenizers")

        backend_with_underscore = find_backend("    if not is_tensorflow_text_available():")
        self.assertEqual(backend_with_underscore, "tensorflow_text")

        double_backend = find_backend("    if not (is_sentencepiece_available() and is_tokenizers_available()):")
        self.assertEqual(double_backend, "sentencepiece_and_tokenizers")

        double_backend_with_underscore = find_backend(
            "    if not (is_sentencepiece_available() and is_tensorflow_text_available()):"
        )
        self.assertEqual(double_backend_with_underscore, "sentencepiece_and_tensorflow_text")

        triple_backend = find_backend(
            "    if not (is_sentencepiece_available() and is_tokenizers_available() and is_vision_available()):"
        )
        self.assertEqual(triple_backend, "sentencepiece_and_tokenizers_and_vision")

    def test_read_init(self):
        objects = read_init()
        # We don't assert on the exact list of keys to allow for smooth grow of backend-specific objects
        self.assertIn("torch", objects)
        self.assertIn("tensorflow_text", objects)
        self.assertIn("sentencepiece_and_tokenizers", objects)

        # Likewise, we can't assert on the exact content of a key
        self.assertIn("BertModel", objects["torch"])
        self.assertIn("TFBertModel", objects["tf"])
        self.assertIn("FlaxBertModel", objects["flax"])
        self.assertIn("BertModel", objects["torch"])
        self.assertIn("TFBertTokenizer", objects["tensorflow_text"])
        self.assertIn("convert_slow_tokenizer", objects["sentencepiece_and_tokenizers"])

    def test_create_dummy_object(self):
        dummy_constant = create_dummy_object("CONSTANT", "'torch'")
        self.assertEqual(dummy_constant, "\nCONSTANT = None\n")

        dummy_function = create_dummy_object("function", "'torch'")
        self.assertEqual(
            dummy_function, "\ndef function(*args, **kwargs):\n    requires_backends(function, 'torch')\n"
        )

        expected_dummy_class = """
class FakeClass(metaclass=DummyObject):
    _backends = 'torch'

    def __init__(self, *args, **kwargs):
        requires_backends(self, 'torch')
"""
        dummy_class = create_dummy_object("FakeClass", "'torch'")
        self.assertEqual(dummy_class, expected_dummy_class)

    def test_create_dummy_files(self):
        expected_dummy_pytorch_file = """# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import DummyObject, requires_backends


CONSTANT = None


def function(*args, **kwargs):
    requires_backends(function, ["torch"])


class FakeClass(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])
"""
        dummy_files = create_dummy_files({"torch": ["CONSTANT", "function", "FakeClass"]})
        self.assertEqual(dummy_files["torch"], expected_dummy_pytorch_file)
