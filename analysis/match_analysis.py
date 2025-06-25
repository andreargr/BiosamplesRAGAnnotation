import json  # use json data
import pandas as pd  # dataframe manipulation
import matplotlib.pyplot as plt  # data visualization
from pathlib import Path

from class_names import df_dash

def data_process(filename):
    """
    Load data from a JSON file and convert it to a structured DataFrame.

    Parameters:
        filename (str): Path to the JSON file containing ontology class data.

    """
    with open(filename, 'r') as archivo:
        dict_classes = json.load(archivo)
    df = pd.DataFrame.from_dict(dict_classes, orient='index')
    new_row = pd.DataFrame([df.columns], columns=df.columns)
    df_process = pd.concat([new_row, df], ignore_index=True)
    df_process.columns = ['CLO_C', 'CLO_M', 'CL_C', 'CL_M', 'UBERON_C', 'UBERON_M', 'BTO_C', 'BTO_M']
    df_process = df_process.drop(0)

    keys = []
    for key in dict_classes.keys():
        keys.append(key)
    df_process['Label'] = keys  # add label column

    df_process.fillna("", inplace=True)  # replace 'Nonetype' values

    return df_process


def match_calculation(type):
    """
    Calculate the perfect match ratio for each ontology by comparing columns for expected and predicted values.

    Parameters:
        type (str): The ontology type ('CL', 'CT', 'A', or 'dash').

    """
    if type == 'CL':
        df = data_process('results_4o_mini/contribution_file_CL.json')
        col_1 = 'CLO'
        col_2 = 'BTO'
    elif type == 'CT':
        df = data_process('results_4o_mini/contribution_file_CT.json')
        col_1 = 'CL'
        col_2 = 'BTO'
    elif type == 'A':
        df = data_process('results_4o_mini/contribution_file_A.json')
        col_1 = 'UBERON'
        col_2 = 'BTO'
    elif type == 'dash':
        return 0  # Set perfect match to 0 for 'dash'
    else:
        raise ValueError("Unrecognized ontology type")

    perfect_match = 0
    no_perfect_match = 0

    control_1 = f'{col_1}_C'
    control_2 = f'{col_2}_C'
    test_1 = f'{col_1}_M'
    test_2 = f'{col_2}_M'

    for index, row in df.iterrows():
        string1a = row[control_1]
        string2a = row[test_1]
        string1b = row[control_2]
        string2b = row[test_2]
        if string1a == string2a and string1b == string2b:
            perfect_match += 1
        else:
            no_perfect_match += 1

    total = perfect_match + no_perfect_match
    index_pm = perfect_match / total
    return index_pm


def calculate_metrics(ontology_type):
    """
    Calculate precision, recall (exhaustiveness), and F1-score for each ontology type.

    Parameters:
        ontology_type (str): The ontology type ('CL', 'CT', 'A', or 'dash').

    Returns:
        tuple: Three dictionaries containing the accuracy (precision), recall (exhaustiveness),
               and F1-score for each ontology.
    """
    if ontology_type == 'CL':
        df = data_process('results_4o_mini/contribution_file_CL.json')
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
    elif ontology_type == 'CT':
        df = data_process('results_4o_mini/contribution_file_CT.json')
        suffixes = ['CL', 'UBERON', 'BTO']
    elif ontology_type == 'A':
        df = data_process('results_4o_mini/contribution_file_A.json')
        suffixes = ['UBERON', 'BTO']
    elif ontology_type == 'dash':
        df = df_dash
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
    else:
        raise ValueError("Unrecognized ontology type")

    precisions = {}
    accuracies = {}
    recall = None if ontology_type == 'dash' else {}
    f1 = None if ontology_type == 'dash' else {}

    for suffix in suffixes:
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        if true_col in df.columns and pred_col in df.columns:
            for i in range(len(df)):
                true_val = df.iloc[i][true_col]
                pred_val = df.iloc[i][pred_col]
                # Skip rows where both true and predicted values are "-"
                if true_val == "-" and pred_val == "-":
                    #print(true_val, pred_val)
                    tn += 1
                elif true_val != "-" and pred_val == "-":
                    #print(true_val, pred_val)
                    fn += 1
                elif true_val == pred_val:
                    #print(true_val, pred_val)
                    tp += 1
                elif true_val != pred_val:
                    #print(true_val,"|||",pred_val)
                    fp += 1
            print(ontology_type, suffix, "TP:", tp, " FP:", fp, " FN:", fn, " TN:", tn)
            # Calculate metrics if not in 'dash' mode
            if ontology_type != 'dash':
                # Calculate recall (exhaustiveness) and F1-score
                recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_value = (2 * tp) / (2 * tp + fn + fp) if ( 2 * tp + fn + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fn + fp)
                recall[suffix] = recall_value
                f1[suffix] = f1_value
                accuracies[suffix] = accuracy

            # Calculate accuracy (precision) based on the evaluated cases (excluding skipped rows)
            total_evaluated = tp + fp
            precision = tp / total_evaluated if total_evaluated > 0 else 0
            precisions[suffix] = precision

        else:
            if ontology_type == 'dash':
                recall[suffix] = None
                f1[suffix] = None
                precisions[suffix] = None
                accuracies[suffix] = None

    return precisions, recall, f1 , accuracies


def plot_combined_metrics(save_dir='.', dpi=300):
    import matplotlib.pyplot as plt
    import pandas as pd

    types = ['CL', 'CT', 'A', 'dash']
    metrics_data = {'precision': {}, 'accuracy': {}, 'exhaustiveness': {}, 'f1_score': {}, 'perfect_match': {}}

    for type in types:
        precision, exhaust, f1, accuracy = calculate_metrics(type)
        metrics_data['precision'][type] = precision
        metrics_data['accuracy'][type] = accuracy
        metrics_data['exhaustiveness'][type] = exhaust
        metrics_data['f1_score'][type] = f1
        metrics_data['perfect_match'][type] = match_calculation(type)

    types_exhaust_f1 = ['CL', 'CT', 'A']

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    # Precision + Perfect Match
    df_precision = pd.DataFrame(metrics_data['precision']).T
    df_precision['Perfect_match'] = metrics_data['perfect_match']
    df_precision.plot(kind='bar', ax=axs[0], width=0.8)
    axs[0].set_title('Precision and Perfect Match', fontsize=16)
    axs[0].set_ylim(0, 1.2)
    axs[0].set_xlabel('Types', fontsize=14)
    axs[0].set_ylabel('Precision', fontsize=14)
    axs[0].tick_params(labelsize=12)
    axs[0].legend(title='Ontologies', fontsize=12, title_fontsize=13)
    for p in axs[0].patches:
        axs[0].annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom',
                        fontsize=11, xytext=(0, 3), textcoords='offset points')

    # Recall
    df_exhaust = pd.DataFrame({key: metrics_data['exhaustiveness'][key] for key in types_exhaust_f1}).T
    df_exhaust.plot(kind='bar', ax=axs[1], width=0.8)
    axs[1].set_title('Recall (Exhaustiveness)', fontsize=16)
    axs[1].set_ylim(0, 1.2)
    axs[1].set_xlabel('Types', fontsize=14)
    axs[1].set_ylabel('Recall', fontsize=14)
    axs[1].tick_params(labelsize=12)
    axs[1].legend(title='Ontologies', fontsize=12, title_fontsize=13)
    for p in axs[1].patches:
        axs[1].annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom',
                        fontsize=11, xytext=(0, 3), textcoords='offset points')

    # F1-score
    df_f1 = pd.DataFrame({key: metrics_data['f1_score'][key] for key in types_exhaust_f1}).T
    df_f1.plot(kind='bar', ax=axs[2], width=0.8)
    axs[2].set_title('F1-score', fontsize=16)
    axs[2].set_ylim(0, 1.2)
    axs[2].set_xlabel('Types', fontsize=14)
    axs[2].set_ylabel('F1-score', fontsize=14)
    axs[2].tick_params(labelsize=12)
    axs[2].legend(title='Ontologies', fontsize=12, title_fontsize=13)
    for p in axs[2].patches:
        axs[2].annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom',
                        fontsize=11, xytext=(0, 3), textcoords='offset points')

    # Accuracy
    df_acc = pd.DataFrame({key: metrics_data['accuracy'][key] for key in types_exhaust_f1}).T
    df_acc.plot(kind='bar', ax=axs[3], width=0.8)
    axs[3].set_title('Accuracy', fontsize=16)
    axs[3].set_ylim(0, 1.2)
    axs[3].set_xlabel('Types', fontsize=14)
    axs[3].set_ylabel('Accuracy', fontsize=14)
    axs[3].tick_params(labelsize=12)
    axs[3].legend(title='Ontologies', fontsize=12, title_fontsize=13)
    for p in axs[3].patches:
        axs[3].annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom',
                        fontsize=11, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()  # → distribuye bien los ejes
    fig.canvas.draw()  # → calcula posiciones reales

    # ----------- NUEVO: guardar cada subgráfica por separado ----------
    nombres_archivos = [
        'precision_perfect_match.png',
        'recall_exhaustiveness.png',
        'f1_score.png',
        'accuracy.png',
    ]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for ax, nombre in zip(axs, nombres_archivos):
        # bounding box del eje, convertido a pulgadas
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()) \
            .transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(Path(save_dir) / nombre, dpi=dpi, bbox_inches=bbox)
    # ------------------------------------------------------------------

    fig.savefig(Path(save_dir) / 'metrics_summary.png', dpi=dpi)
    plt.show()



def main():
    plot_combined_metrics()

if __name__ == "__main__":
    main()
