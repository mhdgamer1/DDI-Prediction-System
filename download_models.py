import os
from huggingface_hub import snapshot_download

REPO_ID = "MhdTarras/ddi-prediction-models"
LOCAL_DIR = "./models"

REQUIRED_FILES = [
    "ddi_allergy_config.pkl",
    "ddi_allergy_evaluation.pkl",
    "ddi_final_model_v2.pkl",
    "ddi_model_metadata_v2.pkl",
    "ddi_pregnancy_data.pkl",
    "ddi_severity_config.pkl",
    "ddi_severity_major.pkl",
    "ddi_severity_minor.pkl",
    "ddi_support_data_v2.pkl",
]


def download_models():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    missing = [
        f
        for f in REQUIRED_FILES
        if not os.path.exists(os.path.join(LOCAL_DIR, f))
    ]

    if not missing:
        print("✅ All DDI models already exist.")
        return

    print("⬇️ Missing models:")
    for f in missing:
        print(f"  - {f}")

    print("⬇️ Downloading DDI models from Hugging Face...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=LOCAL_DIR,
        allow_patterns=REQUIRED_FILES,
    )

    print("✅ DDI models downloaded successfully.")