import json
from pathlib import Path

nb = json.loads(Path("models.ipynb").read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if "BOOSTING_STUDIES = {" not in "".join(cell.get("source", [])):
        continue
    new_source = []
    for line in cell["source"]:
        new_source.append(line)
        if line.strip().startswith("study_name="):
            new_source.append("        storage=OPTUNA_STORAGE,\n")
            new_source.append("        load_if_exists=True,\n")
    cell["source"] = new_source
    break

Path("models.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("done")
