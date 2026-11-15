"""Check public documentation, result tables, provenance and redistribution boundaries."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def main():
    required=['README.md','README.zh-CN.md','LICENSE','NOTICE','CITATION.cff',
              'docs/EXPERIMENTS.md','docs/RESULTS.md','data/README.md',
              'third_party/WM-MS3M-LICENSE','configs/weights_manifest.json']
    for name in required:
        assert (ROOT/name).is_file(),name
    # Raw traces and generated tensors never belong in the source release.
    assert not list((ROOT/'data/raw').glob('*.csv'))
    for excluded in ['reviews','colosseum-near-rt-ric','upstream','.env','docs/manuscript_zh.md','docs/figures']:
        assert not (ROOT/excluded).exists(),excluded
    assert not list(ROOT.rglob('*.cc')) and not list(ROOT.rglob('*.cpp'))
    for path in ROOT.rglob('*.md'):
        body=path.read_text(encoding='utf-8')
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',body):
            if '://' in link or link.startswith('#'):
                continue
            target=(path.parent/link.split('#')[0]).resolve()
            assert target.is_relative_to(ROOT) and target.exists(),f'{path}: {link}'
    manifest=json.loads((ROOT/'results/source_provenance.json').read_text())
    core=hashlib.sha256((ROOT/'src/wm_ms3m/core.py').read_bytes()).hexdigest()
    assert core==manifest['core_sha256']=='05a723ee538cca0374e9823ac08a285406f05b9bde9e8851908c0ade5372d6c1'
    for name,digest in manifest['aggregate_results'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    checked=0
    for path in list((ROOT/'experiments').glob('*.py'))+list((ROOT/'configs').rglob('*.json')):
        text=path.read_text(encoding='utf-8')
        assert not re.search(r'(?:/mnt/[a-z]/Project/|[A-Z]:[/\\]Project[/\\]|C:[/\\]Users[/\\])',text),str(path)
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}',text),str(path)
        checked+=1
    weights=json.loads((ROOT/'configs/weights_manifest.json').read_text())
    for name,record in weights['files'].items():
        assert (ROOT/name).resolve().is_relative_to(ROOT)
        assert re.fullmatch(r'[a-f0-9]{64}',record['sha256'])
        assert record['url'].startswith('https://github.com/NyxFelineLabs/CT-WM/releases/download/')
    print(json.dumps(dict(status='PASS',code_and_config_files_checked=checked,
        model_core_hash_verified=True,result_hashes_verified=True,
        documentation_links_valid=True,raw_data_excluded=True),indent=2))

if __name__=='__main__':main()
