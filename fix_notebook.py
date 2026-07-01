import json

notebook_path = r'c:\Users\ahmed mehrez\Desktop\nlp\2\NLP (1).ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [line.replace('/content/', '') for line in cell['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Fixed paths in notebook.")
