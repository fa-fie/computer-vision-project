from architectures import *
from utils import *
import torch
import os
import tqdm
from torch.utils.data import DataLoader

weights_folder = os.path.join(os.getcwd(), "model")
results_folder = os.path.join(os.getcwd(), "results")

def evaluate_on_dataset(model, dataloader, device, csv_fname) -> float:
    # If a file already exists, instead load the data from there
    if os.path.isfile(csv_fname):
        accuracy = accuracy_from_csv(csv_fname)
        print(f"Loaded from CSV file - Accuracy: {accuracy:.2f}%")
        return accuracy

    model.eval()  # set to eval mode

    test_correct = 0
    test_total = 0
    
    dfs = []
    with torch.no_grad():
        for images, labels, orig_fnames, fnames in tqdm.tqdm(dataloader, desc="Testing"):
            images, labels_device = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            test_total += labels_device.size(0)
            test_correct += (predicted == labels_device).sum().item()

            df_batch = pd.DataFrame({"Filename": orig_fnames, "Attack Filename": fnames, "Label": labels, "Predicted": predicted.cpu()})
            dfs.append(df_batch)

    accuracy = 100 * test_correct / float(test_total)
    print(f"\n===============================")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"===============================")

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(csv_fname, index=False)

    return accuracy


def accuracy_from_csv(csv_fname):
    df = pd.read_csv(csv_fname, index_col="Filename")
    correct_count = len(df[df["Label"] == df["Predicted"]])
    total_count = len(df)
    return 100 * correct_count / float(total_count)

def eval_attack(model, device, attack_name, weights_fname, batch_size=64) -> float:
    dataset = TestDataset(attack_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    print(f"Evaluating model on {attack_name} attack test set")
    csv_path = os.path.join(results_folder, f"{weights_fname}_{attack_name}_test_results.csv")
    return evaluate_on_dataset(model, dataloader, device, csv_path)


def get_accuracies_of_model(weights_fname, attacks=["occlusion", "shadow", "noise_blur", "graffiti"], batch_size=64):
    csv_path = os.path.join(results_folder, weights_fname + "_percentages.csv")

    # If the result file does not already exist, run evaluation first
    if not os.path.isfile(csv_path):
        eval_model_on_test_sets(weights_fname, attacks, batch_size)
    
    # Load and return the data
    df = pd.read_csv(csv_path, index_col="Dataset")
    return {attack_name: attack_acc["Test Acc"] for attack_name, attack_acc in df.to_dict("index").items()}


def eval_model_on_test_sets(weights_fname, attacks=["occlusion", "shadow", "noise_blur", "graffiti"], batch_size=64):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(weights_folder, weights_fname + ".pth"),
            map_location=device,
            weights_only=True,
        )
    )

    # Evaluate on attacks
    accuracies = {attack : eval_attack(model, device, attack, weights_fname, batch_size) for attack in attacks}

    # Evaluate on test set
    print(f"Evaluating model on initial test set")
    csv_path = os.path.join(results_folder, f"{weights_fname}_test_results.csv")
    test_dataloader = DataLoader(TestDataset(), batch_size=batch_size, shuffle=False)
    accuracies["Initial"] = evaluate_on_dataset(model, test_dataloader, device, csv_path)

    # Save the results
    df = pd.DataFrame(columns=["Dataset", "Test Acc"])
    for attack_name, attack_acc in accuracies.items():
        df.loc[len(df)] = [attack_name, attack_acc]
    csv_path = os.path.join(results_folder, weights_fname + "_percentages.csv")
    df.to_csv(csv_path, index=False)

    return accuracies


def eval_model_on_own_imgs(weights_fname, batch_size=64):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(weights_folder, weights_fname + ".pth"),
            map_location=device,
            weights_only=True,
        )
    )

    # Initialize dataset
    dataloader = DataLoader(OwnImagesDataset(), batch_size=batch_size, shuffle=False)

    # Run eval
    csv_path = os.path.join(results_folder, f"{weights_fname}_own_imgs_results.csv")
    evaluate_on_dataset(model, dataloader, device, csv_path)

def find_improved_prediction_imgs(csv_fname_wrong, csv_fname_correct):
    df_wrong = pd.read_csv(os.path.join(results_folder, csv_fname_wrong + "_test_results.csv"))
    df_wrong = df_wrong[df_wrong["Label"] != df_wrong["Predicted"]]

    df_correct = pd.read_csv(os.path.join(results_folder, csv_fname_correct + "_test_results.csv"))
    df_correct = df_correct[df_correct["Label"] == df_correct["Predicted"]]

    return df_wrong.merge(df_correct, left_on="Filename", right_on="Filename", suffixes=("_A", "_B"))


def compare_model_predictions(csv_fname_initial, csv_fnames_to_compare, csv_out_fname):
    init = pd.read_csv(os.path.join(results_folder, csv_fname_initial + "_test_results.csv"))

    total_count = float(len(init))
    get_p = lambda df: round((len(df) / total_count) * 100, 1)

    dict_out = {
        "Model": [],
        "Same (correct)": [],
        "Same (incorrect)": [],
        "Improved": [],
        "Worsened": []
    }

    for csv_comp in csv_fnames_to_compare:
        comp = pd.read_csv(os.path.join(results_folder, csv_comp + "_test_results.csv"))
        merged = init.merge(comp, left_on="Filename", right_on="Filename", suffixes=("_init", "_comp"))
        dict_out["Model"].append(csv_comp)
        dict_out["Same (correct)"].append(get_p(merged[(merged["Label_init"] == merged["Predicted_init"]) & (merged["Label_init"] == merged["Predicted_comp"])]))
        dict_out["Same (incorrect)"].append(get_p(merged[(merged["Label_init"] != merged["Predicted_init"]) & (merged["Label_init"] != merged["Predicted_comp"])]))
        dict_out["Improved"].append(get_p(merged[(merged["Label_init"] != merged["Predicted_init"]) & (merged["Label_init"] == merged["Predicted_comp"])]))
        dict_out["Worsened"].append(get_p(merged[(merged["Label_init"] == merged["Predicted_init"]) & (merged["Label_init"] != merged["Predicted_comp"])]))
    
    df_out = pd.DataFrame(dict_out)
    csv_path = os.path.join(results_folder, csv_out_fname)
    df_out.to_csv(csv_path, index=False)

    return dict_out


if __name__ == "__main__":
    # Compare initial model to adversarially trained models
    # Performance on initial test data
    compare_model_predictions(
        "100_initial_data", 
        [
            "70_initial_30_occlusion", 
            "50_initial_50_occlusion", 
            "30_initial_70_occlusion", 
            "100_initial_100_occlusion",
        ],
        "compare_adv_training_initial.csv"
    )
    # Performance on occlusion test data
    compare_model_predictions(
        "100_initial_data_occlusion", 
        [
            "70_initial_30_occlusion_occlusion", 
            "50_initial_50_occlusion_occlusion", 
            "30_initial_70_occlusion_occlusion", 
            "100_initial_100_occlusion_occlusion"
        ],
        "compare_adv_training_occlusion.csv"
    )