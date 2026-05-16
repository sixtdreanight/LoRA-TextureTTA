"""Extract WDF tar files and reorganize frames."""
import os, sys, tarfile
from pathlib import Path
from tqdm import tqdm

dataset_path = sys.argv[1]  # e.g. F:/数据集/deepfake_in_the_wild

for split_dir in ['real_train', 'real_test', 'fake_train', 'fake_test']:
    src = Path(dataset_path) / split_dir
    if not src.is_dir():
        continue
    tars = sorted(src.glob('*.tar*'))
    for tar_path in tqdm(tars, desc=split_dir):
        try:
            tf = tarfile.open(str(tar_path), 'r')
        except tarfile.ReadError:
            continue
        # Each tar contains: ./<tar_stem>/real|fake/<video_id>/<frame>.png
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith('.png'):
                continue
            parts = member.name.strip('./').split('/')
            # parts: [<tar_stem>, real|fake, <video_id>, <frame>.png]
            if len(parts) < 4:
                continue
            video_id = parts[-2]
            frame_name = parts[-1]
            dst_dir = src / 'frames' / video_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / frame_name
            if not dst.exists():
                tf.extract(member, path=str(src / 'frames'))
                # Move from nested path to flat: frames/<tar_stem>/real|fake/<video_id>/<frame>.png
                # → frames/<video_id>/<frame>.png
                extracted = src / 'frames' / Path(member.name).relative_to('.')
                if extracted != dst:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if extracted.exists():
                        extracted.rename(dst)
        tf.close()
    # Clean up empty intermediate dirs under frames
    frames_root = src / 'frames'
    if frames_root.is_dir():
        for d in sorted(frames_root.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"{split_dir} done, {len(list((src/'frames').rglob('*.png')))} frames")
