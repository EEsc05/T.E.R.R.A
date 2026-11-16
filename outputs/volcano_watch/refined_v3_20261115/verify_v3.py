"""Real browser regressions for map progression, input provenance and applications."""
from pathlib import Path
from datetime import date,timedelta
import hashlib,json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
index=json.loads((ROOT/'dataset.json').read_text());errors=[];requests=[];cases=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
 def open_explore():
  page.goto((ROOT/'explore.html').as_uri());page.wait_for_function('Boolean(window.TerraExplore?.state().stack_id && window.TerraFourBranch?.predict)')
 def generate(expected):
  page.click('#generate-forecast');page.wait_for_function('(n)=>TerraExplore.state().generated.length===n',arg=expected)
  page.wait_for_function('!document.querySelector("#generate-forecast").disabled')
 def select(entry):
  page.select_option('#country',entry['country']);page.select_option('#region',entry['physical_volcano']);page.select_option('#stack',entry['id'])
  page.wait_for_function('(id)=>TerraExplore.state().stack_id===id && !document.querySelector("#selection-status").textContent.includes("Loading")',arg=entry['id'])
 open_explore()
 assert page.locator('#map-empty').is_hidden()
 assert page.locator('#forecast-frames button').count()==2
 for entry in index['datasets']:
  select(entry)
  origin=page.locator('#forecast-start').input_value();target=(date.fromisoformat(origin)+timedelta(days=14)).isoformat()
  page.fill('#forecast-target',target);generate(2)
  state=page.evaluate('TerraExplore.state()');assert state['date']==target
  assert state['sequence_dates']==[origin,(date.fromisoformat(origin)+timedelta(days=7)).isoformat(),target]
  assert state['layer']=='forecast' and page.locator('#map-empty').is_hidden()
  assert all(f['input_counts']=={'observed':6,'predicted':0,'missing':0} for f in state['generated'])
  assert len({tuple(f['input_dates']) for f in state['generated']})==1
  assert page.locator('#forecast-frames button').count()==3
  limits=[]
  for i in range(3):
   page.locator('#forecast-frames button').nth(i).click();assert page.locator('#map-empty').is_hidden()
   limits.append(page.locator('#legend-high').inner_text())
  assert len(set(limits))==1
  page.click('[data-layer="observed"]');page.click('[data-layer="forecast"]');assert page.locator('#map-empty').is_hidden()
  assert 'government resource' in page.locator('#official-resources').inner_text().lower()
  cases.append(entry['id']);print('PASS',entry['id'],flush=True)
 # Default user action creates five direct forecasts plus the real starting map.
 open_explore();generate(5);state=page.evaluate('TerraExplore.state()');origin=state['sequence_dates'][0]
 assert len(state['sequence_dates'])==6
 assert page.evaluate('document.activeElement.id')=='heatmap'
 assert page.locator('#map-kind').inner_text()=='Generated four-branch forecast'
 # Manual play visits only populated maps and ends at the target.
 page.click('#play');seen=[]
 for _ in range(29):
  page.wait_for_timeout(180);assert page.locator('#map-empty').is_hidden();seen.append(page.locator('#map-date').inner_text())
 assert set(state['sequence_dates'][1:]).issubset(set(seen))
 assert page.locator('#map-date').inner_text()==state['sequence_dates'][-1]
 assert page.locator('#play').inner_text()=='▶'
 page.check('#allow-recursive');page.click('#use-prediction');generate(10)
 state=page.evaluate('TerraExplore.state()');assert state['generated'][-1]['input_counts']=={'observed':5,'predicted':1,'missing':0}
 assert state['generated'][-1]['change_origin_date']==origin
 assert origin in page.locator('#skill-copy').inner_text()
 page.evaluate('document.activeElement.blur();window.scrollTo(0,0)');page.screenshot(path=str(ROOT/'preview-explore-desktop.png'),full_page=True)
 # Invalid dates produce a visible explanation, not a silent empty map.
 page.fill('#forecast-target',origin);page.click('#generate-forecast');assert 'Choose a target' in page.locator('#prediction-status').inner_text()
 assert 'could not be generated' in page.locator('#map-feedback').inner_text()
 assert page.locator('#map-empty').is_hidden()
 # No future browsing value may enter a new forecast.
 open_explore();page.evaluate('''()=>{window.savedPredict=TerraFourBranch.predict;TerraFourBranch.predict=async x=>{const y=await savedPredict(x);window.lastPrediction=y.prediction_mm;return y;};}''');generate(5);first=page.evaluate('lastPrediction')
 page.click('#reset-forecast');page.evaluate('()=>{const s=TERRA_STACKS[TerraExplore.state().stack_id];s.observed=s.observed.map(m=>m.map(()=>1e9));}');generate(10);assert first==page.evaluate('lastPrediction')
 # Four actual inputs get two explicitly missing slots.
 page.evaluate('''()=>{const h=TERRA_STACKS[TerraExplore.state().stack_id].causal_history;for(const k of ['dates','maps','valid_masks'])h[k]=[0,2,4,5].map(i=>h[k][i]);}''');page.click('#reset-forecast');generate(15)
 assert page.evaluate('TerraExplore.state().generated.at(-1).input_counts')=={'observed':4,'predicted':0,'missing':2}
 # Every additional activity option gives useful content without fake forecasts.
 open_explore();generate(5)
 kinds=page.locator('#activity-kind option').evaluate_all('(xs)=>xs.map(x=>x.value)')
 for kind in kinds:
  page.select_option('#activity-kind',kind);page.click('#predict-activity');text=page.locator('#activity-result').inner_text()
  assert len(text)>150
  if kind=='deformation':assert '1024 cells' in text
  elif kind!='eruption':assert 'no trained activity model' in text.lower() and 'Additional evidence needed' in text
 # All page/device layouts retain no horizontal overflow.
 for width in [320,390,768,1024,1440]:
  page.set_viewport_size({'width':width,'height':900})
  for name in ['index.html','explore.html','learn.html','about.html']:
   page.goto((ROOT/name).as_uri());page.wait_for_selector('#main-nav',state='attached')
   if name=='explore.html':page.wait_for_function('Boolean(window.TerraExplore?.state().stack_id)')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(name,width)
 page.set_viewport_size({'width':390,'height':844});open_explore();generate(5)
 assert page.locator('#map-empty').is_hidden();assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.evaluate('document.activeElement.blur();window.scrollTo(0,0)');page.screenshot(path=str(ROOT/'preview-explore-mobile.png'),full_page=True)
 page.click('#menu-toggle');page.keyboard.press('Escape');assert page.get_attribute('#menu-toggle','aria-expanded')=='false'
 # Users without reduced motion get automatic playback and a populated final map.
 page.emulate_media(reduced_motion='no-preference');open_explore();generate(5)
 assert page.locator('#play').inner_text()=='Ⅱ';page.wait_for_timeout(4800)
 assert page.locator('#map-kind').inner_text()=='Generated four-branch forecast' and page.locator('#map-empty').is_hidden()
 assert not errors,errors;assert not [u for u in requests if not u.startswith('file:')]
 browser.close()
previous=json.loads((ROOT/'preserved_before.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in previous.items())
result={'all_histories_actual_generated_sequences':cases,'direct_sequence_reuses_identical_causal_inputs':True,'starting_observation_and_all_generated_frames_visible':True,'playback_has_no_empty_observation_only_dates':True,'shared_colour_scale':True,'automatic_and_reduced_motion_behaviour_checked':True,'recursive_inputs_labelled_and_original_date_preserved':True,'explicit_sparse_slots_checked':True,'future_browsing_cube_cannot_affect_predictions':True,'invalid_target_explained_in_generator_and_map_panel':True,'activity_options':kinds,'research_applications_do_not_claim_trained_activity_models':True,'all_four_pages_fit_five_device_widths':True,'browser_errors':errors,'external_runtime_requests':[],'preserved_previous_version_and_engine_files':len(previous)}
with (ROOT/'browser_verification_v3.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
