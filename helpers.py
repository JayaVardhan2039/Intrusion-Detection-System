import json
from pathlib import Path

def load_attack_data():
    file_path = Path(__file__).parent / "attack-data.json"
    with open(file_path, "r") as f:
        return json.load(f)

def get_confidence_based_data(attack_type, confidence):
    data = load_attack_data()
    attack_entries = data.get(attack_type, [])
    
    # Find matching confidence range
    for entry in attack_entries:
        if entry["min_confidence"] <= confidence <= entry["max_confidence"]:
            return {
                "causes": entry.get("causes", []),
                "steps": entry.get("steps", []),
                "explanations": entry.get("explanations", []),
                "precautions": entry.get("precautions", [])
            }
    
    # Fallback to highest confidence if none match
    return attack_entries[-1] if attack_entries else {}