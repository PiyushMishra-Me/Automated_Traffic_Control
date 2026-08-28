from pathlib import Path
import shutil

SOURCE = Path("data/ambulance_source")
OUTPUT = Path("data/ambulance_cleaned")

for split in ["train", "valid", "test"]:
    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"

    output_images = OUTPUT / split / "images"
    output_labels = OUTPUT / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Copy images
    for image_file in source_images.iterdir():
        if image_file.is_file():
            shutil.copy2(image_file, output_images / image_file.name)

    # Convert labels
    for label_file in source_labels.glob("*.txt"):
        new_lines = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                # 0 = Ambulance, 1 = ambulance -> class 0
                if class_id in (0, 1):
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))

                # 2 = siren -> removed

        with open(output_labels / label_file.name, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")

print("Done!")
print(f"Original: {SOURCE}")
print(f"Cleaned:  {OUTPUT}")