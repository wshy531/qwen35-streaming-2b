"""Bind only exact, previously authorized shared regression assets if own copies are absent."""
from pathlib import Path
import sys,json
from selector_qa_common import ROOT,PREP,read,save,media_path

def main():
    if sys.platform=='win32':raise RuntimeError('Bind on cluster; no local media probing')
    plan=read(PREP/'plan.json');original=PREP/'plan_before_media_binding.json'
    if original.exists():raise FileExistsError('Do not repeat path binding')
    save(original,plan);count=0
    for source in plan['sources']:
        if (ROOT/source['relative_path']).is_file():continue
        if source['evaluation_group']!='regression64':raise FileNotFoundError('Long source missing in own dataset')
        source['authorized_shared_readonly_path']='/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos/'+source['name']
        p=media_path(source)
        if not p.is_file():raise FileNotFoundError('Exact authorized regression asset unavailable')
        count+=1
    (PREP/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Bound exact authorized shared regression files read-only:',count,'; no copying/writes to shared directory')

if __name__=='__main__':main()
