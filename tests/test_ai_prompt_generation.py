import os
import unittest
from unittest.mock import patch

import modules.core as core
from modules import chatgpt_utils, service_bootstrap


class AiPromptGenerationTests(unittest.TestCase):
    def tearDown(self):
        core.openai_client = None

    @patch("modules.service_bootstrap._sort_data_files")
    @patch("modules.service_bootstrap.initialize_firebase")
    @patch("modules.service_bootstrap.secretmanager_v1.SecretManagerServiceClient")
    def test_local_startup_initializes_openai_from_environment(
        self, secret_client, _initialize_firebase, _sort_data_files
    ):
        from google.auth.exceptions import DefaultCredentialsError

        secret_client.side_effect = DefaultCredentialsError("no local ADC")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "openai.OpenAI"
        ) as openai:
            service_bootstrap.initialize_services("test-project", "test-bucket")

        openai.assert_called_once_with(api_key="test-key")
        self.assertIs(core.openai_client, openai.return_value)

    def test_missing_client_is_reported_as_failure(self):
        core.openai_client = None

        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            chatgpt_utils._call_openai("Generate a prompt")


if __name__ == "__main__":
    unittest.main()
