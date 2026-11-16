"""Tests for tool-call parsing without external model requests."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from dotenv import load_dotenv
from llm_controller import ModelRequestError, OpenAICompatibleController


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def tool_response(name, arguments):
    return {
        'choices': [{
            'message': {
                'tool_calls': [{
                    'function': {
                        'name': name,
                        'arguments': json.dumps(arguments),
                    }
                }]
            }
        }]
    }


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = OpenAICompatibleController(
            api_key='test-only',
            model='gemini-test',
            base_url='https://example.test/v1beta/openai',
        )

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-only',
        'OPENAI_MODEL': '',
        'LLM_MODEL': '',
        'LLM_BASE_URL': '',
        'OPENAI_BASE_URL': '',
    })
    @patch('dotenv.load_dotenv')
    def test_environment_defaults_to_gpt4(self, _load_dotenv):
        controller = OpenAICompatibleController.from_environment()
        self.assertEqual(controller.model, 'gpt-4')
        self.assertEqual(controller.base_url, 'https://api.openai.com/v1')

    @patch('llm_controller.urllib.request.urlopen')
    def test_search_tool_call_is_normalized_and_uses_supplied_context(self, urlopen):
        urlopen.return_value = Response(tool_response('search', {'query': 'main theorem'}))
        result = self.controller.decide(
            'What does the theorem state?',
            [{'queries': ['theorem'], 'context': 'context text', 'context_ids': ['c1']}],
        )
        self.assertEqual(result, {'action': 'search', 'query': 'main theorem'})
        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode('utf-8'))
        self.assertIn('context text', body['messages'][1]['content'])
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-only')

    @patch('llm_controller.urllib.request.urlopen')
    def test_answer_tool_call_preserves_citations(self, urlopen):
        urlopen.return_value = Response(
            tool_response('answer', {'answer': 'Claim [c1]', 'citations': ['c1']})
        )
        result = self.controller.decide('Question?', [])
        self.assertEqual(
            result,
            {'action': 'answer', 'answer': 'Claim [c1]', 'citations': ['c1']},
        )

    @patch('llm_controller.urllib.request.urlopen')
    def test_malformed_tool_call_is_an_explicit_error(self, urlopen):
        urlopen.return_value = Response({'choices': [{'message': {'content': 'not a tool call'}}]})
        with self.assertRaises(ModelRequestError):
            self.controller.decide('Question?', [])


if __name__ == '__main__':
    unittest.main()
