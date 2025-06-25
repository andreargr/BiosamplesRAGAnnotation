import os
import json

# For langchain and RAG
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def dataframe2prettyjson(dataframe: pd.DataFrame, file: str = None, save: bool = False) -> str:
    """
    Convert a Pandas DataFrame to pretty JSON and optionally save it to a file.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        file (str): The file path to save the pretty JSON.
        save (bool): Whether to save the JSON to a file.

    Returns:
        str: The pretty JSON string representation.
    """
    try:
        json_data = dataframe.to_json(orient='index')
        parsed = json.loads(json_data)
        pretty_json = json.dumps(parsed, indent=4)

        if save and file:
            with open(file, 'w') as f:
                f.write(pretty_json)

        return pretty_json
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {str(je)}")
    except ValueError as ve:
        print(f"Value Error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return ""

def read_txt_file_as_string (file_path): 
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    return file_content

import requests
from dotenv import dotenv_values
import pandas as pd
import json

def load_environment():
    """
    Load environment variables.
    :return: The OPENAI API key.
    """
    config = dotenv_values(dotenv_path="../.env")
    return config.get('BIOPORTAL_API_KEY')

def parse_ontology_acronym_and_local_id(class_id_full_iri):
    """
    Given a full IRI like 'http://purl.obolibrary.org/obo/CLO_0050910',
    infer the ontology acronym (CLO, CL, UBERON, BTO) and the local ID in 'CLO:0050910' format.

    Returns: (ontology_acronym, local_id)
    """
    # Example: http://purl.obolibrary.org/obo/CLO_0050910 -> CLO_0050910
    if "obo/" in class_id_full_iri:
        local_fragment = class_id_full_iri.split("obo/")[-1]  # CLO_0050910
    else:
        # If structured differently, adjust as needed
        local_fragment = class_id_full_iri.split("/")[-1]

    # local_fragment: CLO_0050910
    parts = local_fragment.split("_", 1)  # ["CLO", "0050910"]
    if len(parts) == 2:
        possible_ont = parts[0]  # e.g. "CLO"
        local_id = possible_ont + ":" + parts[1]  # e.g. "CLO:0050910"

        # Validate if possible_ont is actually one of our known acronyms
        if possible_ont in ["CLO", "CL", "UBERON", "BTO"]:
            return possible_ont, local_id
        else:
            # If it's not recognized, just return the entire local_fragment
            return possible_ont, local_fragment
    else:
        return None, None


def get_class_name(ontology_acronym, class_id):
    """
    Retrieve the class prefLabel from BioPortal API for a given class ID.

    Parameters:
        ontology_acronym (str): Ontology acronym (e.g., 'CLO', 'CL', 'UBERON', 'BTO').
        class_id (str): Local identifier for the class within the ontology (e.g. 'CLO:0050910').

    Returns:
        str: The prefLabel of the class if found, or None if the request fails.
    """
    url = f"http://data.bioontology.org/ontologies/{ontology_acronym}/classes/{class_id}"
    api_key = load_environment()  # Replace with your actual API key
    headers = {
        'Authorization': f'apikey token={api_key}',
        'Accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("prefLabel")
    else:
        print("Failed to retrieve data:", response.status_code, class_id)
        return None


# 3) Modify the get_classes function to work with the list of annotations from annotate_text.
def get_classes_from_annotations(annotations_list):
    """
    Takes the list of annotation dictionaries returned by annotate_text(...) and retrieves
    the prefLabel for each class.

    Parameters:
        annotations_list (list): The JSON list returned by the BioPortal Annotator,

    Returns:
        list: A list of dictionaries, each containing:
            {
                "class_id_full_iri": str,
                "ontology_acronym": str or None,
                "local_id": str or None,
                "prefLabel": str or None,
                "match_text": str,
                "match_type": str,
                "from": int,
                "to": int
            }
    """
    results = []

    for ann in annotations_list:
        annotated_class = ann.get("annotatedClass", {})
        class_id_full_iri = annotated_class.get("@id", "")
        annotations_info = ann.get("annotations", [])

        # Parse the full IRI to get acronym + local_id
        acronym, local_id = parse_ontology_acronym_and_local_id(class_id_full_iri)

        # Get the prefLabel via the BioPortal API
        if acronym and local_id:
            label = get_class_name(acronym, local_id)
        else:
            label = None

        # Each 'ann' can contain multiple segments in 'annotations'
        for snippet in annotations_info:
            match_text = snippet.get("text", "")
            match_type = snippet.get("matchType", "")

            # Add to results
            results.append({
                "class_id_full_iri": class_id_full_iri,
                "ontology_acronym": acronym,
                "local_id": local_id,
                "prefLabel": label,
                "match_text": match_text,
                "match_type": match_type,
            })

    return results


# 4) Example usage together with annotate_text (simplified version).
def annotate_text(
    text,
    base_url="http://data.bioontology.org/annotator",
    ontologies="CLO,CL,UBERON,BTO",
    semantic_types=None,
    expand_semantic_types_hierarchy=False,
    expand_class_hierarchy=False,
    class_hierarchy_max_level=0,
    expand_mappings=False,
    stop_words=None,
    minimum_match_length=0,
    exclude_numbers=False,
    whole_word_only=True,
    exclude_synonyms=False,
    longest_only=False
):
    """
    Calls the /annotator endpoint to look for classes in specific ontologies
    (by default CLO, CL, UBERON, BTO) associated with the given text.

    Parameters:
    -----------
    text : str
        The text to be annotated (to search for classes).
    base_url : str
        The base URL of the Annotator service.
    ontologies : str
        A comma-separated list of ontologies (e.g. "CLO,CL,UBERON,BTO").
    semantic_types : str
        A comma-separated list of semantic_types (e.g. "T047,T191").
    expand_semantic_types_hierarchy : bool
        Whether to expand the semantic type hierarchy.
    expand_class_hierarchy : bool
        Whether to expand the class hierarchy (ancestors).
    class_hierarchy_max_level : int
        The maximum level of class hierarchy to be included in the annotation.
    expand_mappings : bool
        Whether to include manual mappings (UMLS, REST, CUI, OBOXREF).
    stop_words : str
        A comma-separated list of stop words (e.g. "the,of,for"), if you want to customize.
    minimum_match_length : int
        The minimum match length.
    exclude_numbers : bool
        Whether to exclude numbers in the text.
    whole_word_only : bool
        Whether only whole-word matches should be considered.
    exclude_synonyms : bool
        Whether synonyms should be excluded.
    longest_only : bool
        Whether only the longest match for each text fragment should be returned.

    Returns:
    --------
    list
        A list of annotations found, where each annotation includes basic information
        of the identified class.
    """
    api_key = load_environment()
    # Prepare query parameters
    params = {
        "text": text,
        "ontologies": ontologies,
        "expand_semantic_types_hierarchy": str(expand_semantic_types_hierarchy).lower(),
        "expand_class_hierarchy": str(expand_class_hierarchy).lower(),
        "class_hierarchy_max_level": class_hierarchy_max_level,
        "expand_mappings": str(expand_mappings).lower(),
        "minimum_match_length": minimum_match_length,
        "exclude_numbers": str(exclude_numbers).lower(),
        "whole_word_only": str(whole_word_only).lower(),
        "exclude_synonyms": str(exclude_synonyms).lower(),
        "longest_only": str(longest_only).lower()
    }

    # If semantic_types is specified
    if semantic_types:
        params["semantic_types"] = semantic_types

    # If custom stop_words are specified
    if stop_words:
        params["stop_words"] = stop_words

    headers = {
        'Authorization': f'apikey token={api_key}',
        'Accept': 'application/json'
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def get_annotations(label):
    """
    Retrieves annotations for a given label using annotator functions.

    Args:
        label (str): The label to annotate.

    Returns:
        list: List of classes with their labels.
    """
    annotations = annotate_text(label)
    classes_with_labels = get_classes_from_annotations(annotations)
    return classes_with_labels


def group_local_ids(classes_with_labels, ontologies):
    """
    Groups local_id and prefLabel by their ontology acronym.

    Args:
        classes_with_labels (list): List of classes with their labels.
        ontologies (list): List of ontology acronyms to include as columns.

    Returns:
        dict: Dictionary with ontologies as keys and lists of dictionaries containing local_id and prefLabel as values.
    """
    local_ids = {ont: [] for ont in ontologies}

    for entry in classes_with_labels:
        acronym = entry.get('ontology_acronym')
        local_id = entry.get('local_id')
        pref_label = entry.get('prefLabel')
        if acronym in local_ids:
            local_ids[acronym].append({
                "local_id": local_id,
                "prefLabel": pref_label
            })
        else:
            # Handle ontologies not previously defined if necessary
            local_ids[acronym] = [{
                "local_id": local_id,
                "prefLabel": pref_label
            }]

    return local_ids


def select_local_id(acronym, entries, label):
    """
    Allows the user to select a local_id when there are multiple options.
    Automatically selects if all local_ids are identical.

    Args:
        acronym (str): Ontology acronym.
        entries (list): List of dictionaries with local_id and prefLabel.
        label (str): The current label being processed.

    Returns:
        str or None: The selected local_id or None if there are no options.
    """
    if not entries:
        print(f"The ontology '{acronym}' has no associated local_id.")
        return None
    elif len(entries) == 1:
        selected = entries[0]['local_id']
        pref_label = entries[0]['prefLabel'] or 'No prefLabel'
        print(
            f"Ontology '{acronym}' has only one local_id: {selected} - {pref_label}. Automatically selected.")
        return selected
    else:
        # Check if all local_ids are identical
        unique_ids = set(entry['local_id'] for entry in entries)
        if len(unique_ids) == 1:
            selected = entries[0]['local_id']
            pref_label = entries[0]['prefLabel'] or 'No prefLabel'
            print(
                f"All local_ids for ontology '{acronym}' are identical: {selected} - {pref_label}. Automatically selected.")
            return selected
        else:
            print(f"\nFor the Label '{label}', the ontology '{acronym}' has multiple local_id:")
            for idx, entry in enumerate(entries, start=1):
                lid = entry['local_id']
                plabel = entry['prefLabel'] or 'No prefLabel'
                print(f"{idx}. {lid} - {plabel}")
            while True:
                try:
                    selection = int(
                        input(f"Select the number corresponding to the local_id you want for '{acronym}': "))
                    if 1 <= selection <= len(entries):
                        return entries[selection - 1]['local_id']
                    else:
                        print(f"Please enter a number between 1 and {len(entries)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")


def select_local_ids(local_ids, label):
    """
    Iterates over ontologies and selects the corresponding local_ids.

    Args:
        local_ids (dict): Dictionary with ontologies and their local_ids.
        label (str): The current label being processed.

    Returns:
        dict: Dictionary with ontologies and the selected local_ids.
    """
    selected_local_ids = {}

    for acronym, entries in local_ids.items():
        selected_id = select_local_id(acronym, entries, label)
        selected_local_ids[acronym] = selected_id

    return selected_local_ids


def create_dataframe(label, type_value, selected_local_ids, ontologies):
    """
    Creates a pandas DataFrame from the provided data.

    Args:
        label (str): The label to include in the DataFrame.
        type_value (str): The type to include in the DataFrame.
        selected_local_ids (dict): Dictionary with ontologies and the selected local_ids.
        ontologies (list): List of ontology acronyms to include as columns.

    Returns:
        pandas.DataFrame: The created DataFrame.
    """
    data = {
        "Label": [label],
        "Type": [type_value],
    }

    for ont in ontologies:
        data[ont] = [selected_local_ids.get(ont)]

    df = pd.DataFrame(data)
    return df


def process_labels_from_tsv(tsv_path, ontologies, output_path=None):
    """
    Processes multiple labels from a TSV file and accumulates the results into a DataFrame.

    Args:
        tsv_path (str): Path to the input TSV file.
        ontologies (list): List of ontology acronyms to include as columns.
        output_path (str, optional): Path to save the resulting DataFrame as a CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: The accumulated DataFrame with all labels.
    """
    # Read the TSV file
    try:
        input_df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return None

    # Initialize an empty list to collect DataFrames
    accumulated_data = []

    # Iterate through each row in the TSV
    for index, row in input_df.iterrows():
        label = row.get('Label')
        type_value = row.get('Type')

        # Handle missing Label or Type
        if pd.isna(label) or pd.isna(type_value):
            print(f"Row {index + 1} is missing Label or Type. Skipping.")
            continue

        print(f"\nProcessing Label: '{label}' with Type: '{type_value}'")

        # 1) Get annotations
        classes_with_labels = get_annotations(label)

        # Optional: Print the JSON of classes with labels
        json_str = json.dumps(classes_with_labels, indent=2, ensure_ascii=False)
        print("Obtained Annotations:")
        print(json_str)

        # 2) Group local_ids by ontology
        local_ids = group_local_ids(classes_with_labels, ontologies)

        # 3) Select local_ids
        selected_local_ids = select_local_ids(local_ids, label)

        # 4) Create the DataFrame for the current label
        df = create_dataframe(label, type_value, selected_local_ids, ontologies)

        # Append to the accumulated data
        accumulated_data.append(df)

    if not accumulated_data:
        print("No data processed. Exiting.")
        return None

    # Concatenate all DataFrames
    final_df = pd.concat(accumulated_data, ignore_index=True)

    # Optional: Save the final DataFrame to a CSV file
    if output_path:
        try:
            final_df.to_csv(output_path, index=False)
            print(f"\nFinal DataFrame saved to '{output_path}'.")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")

    return final_df

