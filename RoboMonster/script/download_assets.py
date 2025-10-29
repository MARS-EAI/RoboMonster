from huggingface_hub import snapshot_download
import os

snapshot_download(repo_id='martelzhang/RoboMonster_assets',
                  local_dir='./assets',
                  repo_type='dataset',
                  resume_download=True)