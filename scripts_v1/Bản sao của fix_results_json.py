import os

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and l.strip() != ']']
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"✅ Cleaned {path}")

if __name__ == "__main__":
    for fname in os.listdir('output'):
        if fname.startswith("result_") and fname.endswith(".json"):
            clean_file(os.path.join("output", fname))
