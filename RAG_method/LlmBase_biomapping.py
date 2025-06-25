import openai
from dotenv import dotenv_values
import json

def load_environment():
    """
    Load environment variables.
    :return: The OPENAI API key.
    """
    config = dotenv_values(dotenv_path="../.env")
    return config.get('OPENAI_API_KEY')

# Define the API key of OpenAI
openai.api_key=load_environment()

class GptLLM:
    def __init__(self, model_name, temperature):
        if model_name not in {
            'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o'
        }:
            raise Exception('Invalid model')

        self.model = model_name
        self.temperature = temperature
        self.responses = []

    def run_inference(self, chat_template):
        completion = openai.chat.completions.create(
            model=self.model,
            messages=chat_template,
            max_tokens=4096,
            temperature=self.temperature,
        )

        response = completion.choices[0].message.content
        self.responses.append({"input": chat_template, "response": response})
        return response

    def get_response(self):
        return self.responses

    def save_results(self, results, name):
        """
        Save the output_reduced_ontologies of the model in JSON format.

        Parameters:
            results (dict): Dictionary containing the label and its correspondings identifiers for each of the ontologies of interest.
            name (str): Name for the new JSON file.
        """
        with open(name, 'w') as file:
            json.dump(results, file, indent=4)


    def save_responses(self, outputpath):
        responses = [entry["response"] for entry in self.responses]
        with open(outputpath, "w", encoding="utf-8") as file:
            file.write("\n".join(responses))

    def clean_responses(self):
        self.responses = []


