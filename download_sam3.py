import os
from modelscope import snapshot_download
from pathlib import Path

def download_sam3():
    model_dir = Path("/home/wangyankun/models/sam3")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting download SAM3 to {model_dir}...")
    
    # Using ModelScope for faster download in China
    # Model ID based on user config path pattern
    # facebook/sam3
    download_path = snapshot_download(
        'facebook/sam3',
        local_dir=str(model_dir),
        allow_file_pattern=['sam3.pt', 'bpe_simple_vocab_16e6.txt.gz']
    )
    
    print(f"Download complete! Files are in: {download_path}")
    print("Files list:")
    for f in model_dir.glob("*"):
        print(f" - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    download_sam3()
