"""Package this new website version without overwriting previous archives."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)
previous=json.loads((ROOT/'preserved_before.json').read_text())
assert all(sha(REPO/p)==h for p,h in previous.items())
model=json.loads((ROOT/'assets/model-four-branch.js').read_text().removeprefix('window.TERRA_FOUR_BRANCH_MODEL=').rstrip(';\n'))
checkpoint=REPO/'outputs/nisar_aria_sakurajima_20261115/models/four_branch.joblib'
assert sha(checkpoint)==model['checkpoint_sha256']
assert json.loads((ROOT/'browser_verification_v3.json').read_text())['browser_errors']==[]
assert all(sha(ROOT/name)==value for name,value in json.loads((ROOT/'final_progression_verification.json').read_text())['final_runtime_sha256'].items())
assert sha(ROOT/'forecast-engine.js')==sha(REPO/'outputs/volcano_watch/refined_v2_20261115/forecast-engine.js')
assert sha(ROOT/'assets/model-four-branch.js')==sha(REPO/'outputs/volcano_watch/refined_v2_20261115/assets/model-four-branch.js')
files=[p for pattern in ('*.html','*.css','*.js') for p in ROOT.glob(pattern)]
files += [p for folder in ('assets','data') for p in (ROOT/folder).rglob('*') if p.is_file()]
files += [ROOT/name for name in ('README.md','engine_README.md','government_sources.md','government_resources.json','dataset.json','activity_sources.json','causal_input_provenance.json','browser_verification_v3.json','final_progression_verification.json','engine_verification.json','engine_sparse_recursive_verification.json','engine_recursive_browser_feedback_verification.json','engine_contract_verification.json')]
files=sorted(set(files))
write('site_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
files.append(ROOT/'site_sha256.json')
manifest={'format':'terra-website-v3-local','checkpoint_sha256':sha(checkpoint),'prior_site_and_engine_hashes_preserved':len(previous),'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
write('package_manifest.json',manifest);files.append(ROOT/'package_manifest.json')
archive=ROOT/'TERRA_website_v3.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in files:z.write(p,p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert set(z.namelist())=={str(p.relative_to(ROOT)) for p in files}
 for name,entry in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
write('package_verification.json',{'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),'crc_and_all_entry_hashes_passed':True,'prior_site_and_engine_hashes_preserved':len(previous),'current_checkpoint_unchanged':True})
print(json.dumps(json.loads((ROOT/'package_verification.json').read_text()),indent=2))
