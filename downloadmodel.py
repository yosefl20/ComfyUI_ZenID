from huggingface_hub import hf_hub_download
import gdown
import os


os.makedirs("../models/instantid", exist_ok=True)

hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="../models/instantid"
)

hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="../models/controlnet",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="../models/controlnet",
)

os.makedirs("../models/insightface/", exist_ok=True)
gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output="./models/", quiet=False, fuzzy=True)

os.system("unzip ../models/insightface/models/antelopev2.zip -d ./models/")