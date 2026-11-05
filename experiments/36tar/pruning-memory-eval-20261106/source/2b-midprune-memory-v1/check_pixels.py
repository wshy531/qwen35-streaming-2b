import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def canvas_ast(p):
    return ast.dump(next(n for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='canvas'),include_attributes=False)
assert canvas_ast(R/'experiments/2b-perception-interference-v1/audit_media.py')==canvas_ast(Path(__file__).with_name('frozen_pixels.py'))
print(json.dumps(dict(passed=True,original_preprocessing_function_AST_identical=True)))
