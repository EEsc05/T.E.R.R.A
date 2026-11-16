from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parent;errors=[]
with sync_playwright() as p:
 b=p.chromium.launch(args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100},reduced_motion='reduce');page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((root/'explore.html').as_uri());page.wait_for_function('Boolean(window.TerraExplore?.state().stack_id)');page.click('#generate-forecast');page.wait_for_function('TerraExplore.state().generated.length===5');page.wait_for_function('!document.querySelector("#generate-forecast").disabled');first=page.evaluate('TerraExplore.state().sequence_dates')
 page.evaluate('document.activeElement.blur();window.scrollTo(0,0)');page.screenshot(path=str(root/'preview-explore-desktop.png'),full_page=True)
 page.check('#allow-recursive');page.click('#use-prediction');page.click('#generate-forecast');page.wait_for_function('TerraExplore.state().generated.length===10');page.wait_for_function('!document.querySelector("#generate-forecast").disabled');state=page.evaluate('TerraExplore.state()')
 assert state['sequence_dates'][:6]==first and len(state['sequence_dates'])==11
 assert page.locator('#forecast-frames button').count()==11 and page.locator('#map-empty').is_hidden()
 assert state['generated'][-1]['input_counts']=={'observed':5,'predicted':1,'missing':0}
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 for i in [0,5,10]:page.locator('#forecast-frames button').nth(i).click();assert page.locator('#map-empty').is_hidden()
 page.evaluate('document.activeElement.blur();window.scrollTo(0,0)');page.screenshot(path=str(root/'preview-explore-mobile.png'),full_page=True)
 assert not errors,errors;b.close()
result={'recursive_series_keeps_all_earlier_prediction_dates':True,'sequence_frames_after_two_generations':11,'recursive_counts':state['generated'][-1]['input_counts'],'populated_first_middle_and_final_maps':True,'mobile_recursive_series_no_overflow':True,'browser_errors':errors,'final_runtime_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['explore.html','explore.js','style.css','forecast-engine.js','assets/model-four-branch.js']}}
with (root/'final_progression_verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
