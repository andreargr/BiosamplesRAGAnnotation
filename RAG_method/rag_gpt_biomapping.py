import argparse
from utils_biomapping import dataframe2prettyjson, read_txt_file_as_string, annotate_text, get_classes_from_annotations
import pandas as pd
import LlmBase_biomapping
import json

class LlmIRIGpt(LlmBase_biomapping.GptLLM):
    name = 'ontology'

    def __init__(self, model_name, json_data, temperature=0.5):
        super().__init__(model_name, temperature)
        self.prompt_path = "./IRIsearch_instructions.txt"
        self.system_input = """As an expert ontology mapper, I need your help in searching for suitable identifiers in the ontologies with the acronyms CLO, CL, UBERON, and BTO, from a provided label."""
        self.json_data = json_data
        self.prompt = ""

    def set_prompt(self, label,identifiers_examples):
        initial_prompt = read_txt_file_as_string(self.prompt_path)
        initial_prompt = initial_prompt.format(label=label, identifiers_examples=identifiers_examples)
        conversation = []
        conversation.append({"role": "system", "content": self.system_input})
        conversation.append({"role": "user", "content": initial_prompt}) 
        self.prompt = conversation     

    def get_prompt(self): 
        return self.prompt

    def clean_responses(self):
        super().clean_responses()


def main(args):
    output_file = args.output_file
    dataset_path = args.dataset_path
    model_name = args.model_name

    print(f"Processing file : {dataset_path}")
    print(f"Selected model : {model_name}")

    dataframe = pd.read_csv(dataset_path, sep='\t',header=0) #tsv file loaded
    dataframe.columns = ['Label', 'CLO', 'CL', 'UBERON', 'BTO', 'Type']
    json_string = dataframe2prettyjson(dataframe) #tsv2json
    json_data = json.loads(json_string)

    print('---------------------------------------------------')
    llm_ontology = LlmIRIGpt(model_name, json_data) #Main task: Obtain the identifiers for each label

    results_dict={}
    for key, value in json_data.items():
        label = value.get("Label")  # Get the label
        annotations = annotate_text(label)
        identifiers_examples = get_classes_from_annotations(annotations)
        print(f"Extracted RAG identifiers for {label}: {identifiers_examples}")
        llm_ontology.set_prompt(label,identifiers_examples) #Prompt generation for the second task (based few-shot)
        response = llm_ontology.run_inference(llm_ontology.prompt) #Get the annotation for each ontology
        results_dict[label] = response
    llm_ontology.save_results(results_dict,output_file)
    print(f"Output file : {output_file}")
    llm_ontology.clean_responses()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file",  type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_name", type=str)
    
    args = parser.parse_args()
    main(args)
