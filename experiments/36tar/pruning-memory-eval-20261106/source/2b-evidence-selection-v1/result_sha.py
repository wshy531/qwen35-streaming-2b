"""Small raw-result checksum inventory, no recursive access."""
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1])
print(json.dumps({n:dict(bytes=(p/n).stat().st_size,sha256=hashlib.sha256((p/n).read_bytes()).hexdigest()) for n in ('answers.jsonl','completed.json')}))
