from huggingface_hub import hf_hub_download
import shutil, os

os.makedirs("data", exist_ok=True)
path = hf_hub_download(
    repo_id="NeuronZero/Symptom2Disease",
    filename="Symptom2Disease.csv",
    repo_type="dataset",
)
shutil.copy(path, "data/Symptom2Disease.csv")
print("Saved to data/Symptom2Disease.csv")