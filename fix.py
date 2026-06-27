import os

repo_path = r"c:\xampp2\htdocs\waqf\waqf-backend"
for root, dirs, files in os.walk(repo_path):
    if ".git" in root:
        continue
    for file in files:
        if file.endswith(('.py', '.md', '.toml', '.txt', '.ini', '.yml')):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'Waqf Backend' in content:
                    new_content = content.replace('Waqf Backend', 'WaqfBackend')
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed {path}")
            except Exception as e:
                print(f"Failed to process {path}: {e}")
