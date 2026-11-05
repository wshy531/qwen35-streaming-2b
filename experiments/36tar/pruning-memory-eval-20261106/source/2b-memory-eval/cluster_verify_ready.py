"""Verify offline 2B assets, imports and tokenizer; never load model weights into RAM."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    project=args.project.resolve()
    if str(project)!='/home/share/czy/swt/my_pape/qwen35-streaming-2b-36tar':
        raise ValueError('Project boundary mismatch')
    output=args.out.resolve()
    if not output.is_relative_to(project) or output.exists():
        raise ValueError('Choose a new output within the authorized project')
    asset_script=project/'repo/experiments/36tar/streaming-learning/code/qwen35-streaming-s4/scripts/model_assets.py'
    spec=importlib.util.spec_from_file_location('pinned_model_assets',asset_script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_dir=project/'models/Qwen3.5-2B'
    manifest=module.verify_assets(model_dir)
    if manifest['model']!='Qwen/Qwen3.5-2B':
        raise ValueError('Expected 2B model identity')
    import torch
    import torchvision
    import accelerate
    import av
    import numpy
    import psutil
    from transformers import AutoProcessor
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization')
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    token_ids=processor.tokenizer.encode('Describe the video.',add_special_tokens=False)
    if not token_ids or torch.cuda.is_initialized():
        raise RuntimeError('Tokenizer or CPU-only contract failed')
    versions={name:importlib.metadata.version(name) for name in
              ('torch','torchvision','transformers','accelerate','av','numpy','psutil','Pillow','huggingface-hub','tokenizers')}
    result=dict(status='offline_assets_imports_tokenizer_verified_no_model_inference',
        model=manifest['model'],revision=manifest['revision'],asset_files=len(manifest['files']),
        asset_bytes=sum(x['bytes'] for x in manifest['files']),
        python=platform.python_version(),platform=platform.platform(),versions=versions,
        compiled_cuda=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized(),
        processor_class=type(processor).__name__,tokenizer_class=type(processor.tokenizer).__name__,
        sample_token_count=len(token_ids),process_rss_bytes=psutil.Process().memory_info().rss,
        code_commit=subprocess.check_output(['git','-C',str(project/'repo'),'rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','-C',str(project/'repo'),'branch','--show-current'],text=True).strip(),
        limitations=['No model was instantiated or loaded into RAM.',
                     'No GPU allocation or driver compatibility verification.',
                     'CPU model regressions and true 2B inference remain pending.'])
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
