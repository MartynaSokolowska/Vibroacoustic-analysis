from segments import *
from pathlib import Path
import os
from pydub import AudioSegment

def split_wav_by_material(movement_type: str, layout_type: str, input_file: str):
    basename = Path(input_file).stem
    audio = AudioSegment.from_wav(input_file)

    try:
        segments = segmentations[movement_type][layout_type]
    except KeyError:
        raise ValueError("Invalid movement_type or layout_type")

    for material in segments:
        for i, (start, end) in enumerate(material.segments, start=1):
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            folder_name = f"results/{material.material_name.replace('/', '_').replace(' ', '_')}"
            os.makedirs(folder_name, exist_ok=True)

            output_path = f"{folder_name}/{basename}_segment{i}.wav"
            audio[start_ms:end_ms].export(output_path, format="wav")
            print(f"Saved: {output_path}")


def split_all():
    zucchini_top_path = Path("data\\01\\chicken-zuchini-experiments_07-10_March_2025\\ChickenBottom_ZuchiniTop")
    zucchini_bottom_path = Path("data\\01\\chicken-zuchini-experiments_07-10_March_2025\\ChickenTop_ZuchiniBottom")

    paths = [
        (zucchini_top_path, "zucchini_top"),
        (zucchini_bottom_path, "zucchini_bottom")
    ]

    for base_path, layout_type in paths:
        for speed_dir in base_path.glob("speed_*"):
            movement_type = "slow"

            speed_mapping = {
                "speed_10": "slow",
                "speed_15": "medium",
                "speed_25": "fast"
            }

            movement_type = speed_mapping.get(speed_dir.name, "slow")
            for wav_file in speed_dir.glob("*.wav"):
                split_wav_by_material(movement_type, layout_type, str(wav_file))


split_all()