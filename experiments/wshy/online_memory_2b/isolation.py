"""Deployment write boundaries, established before third-party imports."""
import os
from pathlib import Path
import sys
import tempfile


def write_root():
    approved = Path('D:/AI_project/projects/streaming_video_wsy' if os.name == 'nt'
                    else '/home/share/czy/wsy')
    requested = Path(os.environ.get('QWEN_WRITE_ROOT', str(approved)))
    if (not requested.is_absolute() or not requested.is_dir()
            or requested.resolve() != approved or approved.resolve() != approved):
        raise ValueError('QWEN_WRITE_ROOT must be the confirmed local or cluster boundary')
    return requested.resolve()


def contained(path, root):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('path escapes the approved write boundary: ' + str(path))
    return path


def output_path(path):
    return contained(path, write_root())


def configure():
    root = write_root()
    repo = Path(__file__).resolve().parents[3]
    contained(repo, root)
    git_dir = repo / '.git'
    if not git_dir.is_dir() or git_dir.is_symlink():
        raise ValueError('an independent repository with internal .git is required')
    contained(git_dir, root)
    if (git_dir / 'objects/info/alternates').exists() or (git_dir / 'commondir').exists():
        raise ValueError('external Git object or common-directory references are forbidden')
    cache = contained(os.environ.get('QWEN_CACHE_ROOT', root / 'cache/online-memory-2b'), root)
    locations = {
        'TMPDIR': cache / 'tmp', 'TMP': cache / 'tmp', 'TEMP': cache / 'tmp',
        'XDG_CACHE_HOME': cache / 'xdg', 'XDG_CONFIG_HOME': cache / 'config',
        'XDG_DATA_HOME': cache / 'data', 'XDG_STATE_HOME': cache / 'state',
        'HF_HOME': cache / 'hf',
        'HF_HUB_CACHE': cache / 'hf/hub', 'HUGGINGFACE_HUB_CACHE': cache / 'hf/hub',
        'HF_ASSETS_CACHE': cache / 'hf/assets', 'HF_DATASETS_CACHE': cache / 'hf/datasets',
        'PIP_CACHE_DIR': cache / 'pip', 'TORCH_HOME': cache / 'torch',
        'TORCH_EXTENSIONS_DIR': cache / 'torch_extensions',
        'TORCHINDUCTOR_CACHE_DIR': cache / 'inductor',
        'PYTORCH_KERNEL_CACHE_PATH': cache / 'torch_kernels',
        'TRITON_CACHE_DIR': cache / 'triton',
        'CUDA_CACHE_PATH': cache / 'cuda', 'NUMBA_CACHE_DIR': cache / 'numba',
        'MPLCONFIGDIR': cache / 'matplotlib',
    }
    # Validate every target before the first mkdir, including existing symlinks.
    locations = {key: contained(path, root) for key, path in locations.items()}
    for path in set(locations.values()):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(path) for key, path in locations.items()})
    # Remove the obsolete HF override so it cannot supersede the guarded paths.
    os.environ.pop('TRANSFORMERS_CACHE', None)
    os.environ.update(QWEN_WRITE_ROOT=str(root), QWEN_CACHE_ROOT=str(cache),
                      PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', GIT_OPTIONAL_LOCKS='0')
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(locations['TMPDIR'])
    return root
