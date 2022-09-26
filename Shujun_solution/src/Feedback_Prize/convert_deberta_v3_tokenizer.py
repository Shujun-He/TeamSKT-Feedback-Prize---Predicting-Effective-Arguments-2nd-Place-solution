

# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers
import shutil
from pathlib import Path

transformers_path = Path("/home/exx/.conda/envs/torch/lib/python3.8/site-packages/transformers")

input_dir = Path("../../input/deberta_v3_tokenizer_fast/")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename

    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)
