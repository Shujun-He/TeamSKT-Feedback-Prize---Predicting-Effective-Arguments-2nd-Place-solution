import os

current_dir=os.getcwd().split('/')[-1]
above_dir=os.getcwd().split('/')[-2]
title=f"fb_{above_dir}_{current_dir}"

slug=f"""
{{
  "title": \"{title}\",
  "id": "shujun717/{title.replace('_','-')}",
  "licenses": [
    {{
      "name": "CC0-1.0"
    }}
  ]
}}"""

with open('models/dataset-metadata.json','w+') as f:
    f.write(slug)
