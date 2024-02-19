import octoai
from octoai.errors import OctoAIError
import os

REPEAT_REQUEST_TO_OCTOAI_SERVER = 10

class OctoAIEndpointLM:
    def __init__(
        self,
        url: str=None,
        model_name: str=None,
        ):
        self.token = os.environ["OCTOAI_TOKEN"]
        if self.token is None:
            raise ValueError("TOKEN not found.")

        if url is not None:
            self.endpoint = url
        else:
            self.endpoint = os.environ["ENDPOINT"]
            if self.endpoint is None:
                raise ValueError("ENDPOINT not found.")

        self.model_name = model_name
        if self.model_name is None:
            raise ValueError("model_name not found.")

        self.system_prompt = None

        self.runner_args = {
        "model": self.model_name,
        "stream": False,
        "stop": None,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        }

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.runner_args["messages"] = [
            {
                "role": "system",
                "content": prompt,
            }
        ]

    def model_generate(self, message: str) -> str:
        success = False
        for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
            response = self.run_model(message)
            if response is not None:
                success = True
                break

        if success:
            return self.get_result(response)
        else:
            print("ERROR: response check failed. Dummy response was inserted")
            return self.dummy_result()

    def run_model(self, message: str) -> dict:
        request = self.runner_args
        request["messages"] = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": message,
            }
        ]

        client = octoai.client.Client(
            token=self.token,
        )
        try:
            response = client.infer(
                endpoint_url=self.endpoint + "/v1/chat/completions",
                inputs=request,
            )
            return response
        except OctoAIError:
            return None

    def dummy_result(self):
        return "Dummy response"

    def get_result(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]
    