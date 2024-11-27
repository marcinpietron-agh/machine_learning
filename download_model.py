
import os
from huggingface_hub import snapshot_download

os.environ["TRANSFORMERS_CACHE"] = "/net/tscratch/people/plgpietron/llama"

from huggingface_hub import hf_hub_download
#hf_hub_download(repo_id="meta-llama/Llama-2-13b-chat-hf") #, filename="config.json")


os.environ["TRANSFORMERS_CACHE"] = "/net/tscratch/people/plgpietron/llama"
#snapshot_download(repo_id="meta-llama/Llama-2-13b-hf", local_dir="/net/tscratch/people/plgpietron/llama")
snapshot_download(repo_id="speakleash/Bielik-11B-v2.2-Instruct", local_dir="/net/tscratch/people/plgpietron/llama", token="")