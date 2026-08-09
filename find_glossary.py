import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

scripts = re.findall(r'<script[^>]*>(.*?)</script>', content, re.S)
print(f"Total script tags: {len(scripts)}")
for i, s in enumerate(scripts):
    print(f"Script {i}: {len(s)} chars, starts with: {s[:100]}")

# Find the glossary script (the one with TECH_GLOSSARY)
for i, s in enumerate(scripts):
    if 'TECH_GLOSSARY' in s:
        print(f"\nGlossary script is at index {i}, length {len(s)}")
        break