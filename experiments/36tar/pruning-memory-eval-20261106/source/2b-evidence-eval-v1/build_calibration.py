"""Single-assistant reviewed claims from three actual sampled contact sheets.

Not independent annotation, and not a representative benchmark. Never load QA scores.
"""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
rows=[
 ('2591888090','What color is the stroller canopy?','Red.','The stroller canopy is red.','supported','sufficient'),
 ('2591888090','Who is visibly pushing the stroller?','A small child.','A small child has hands on the stroller handles while walking behind it.','supported','sufficient'),
 ('2591888090','Where is the stroller moving?','On a sidewalk beside grass.','The stroller wheels are on a paved sidewalk with grass beside it.','supported','sufficient'),
 ('10008553263','What does the foreground performer hold near the mouth?','A microphone.','A handheld microphone is near the foreground performer\'s mouth.','supported','sufficient'),
 ('10008553263','What color is the cloth over the performer\'s shoulders?','Red.','A red cloth is draped over the performer\'s shoulders.','supported','sufficient'),
 ('10008553263','What instrument is visible in the background near the end?','A drum kit.','A drum kit is visible behind the foreground performer near 12-14 seconds.','supported','sufficient'),
 ('10315129403','Where are the two foreground people?','In an indoor corridor.','Two people are in a corridor with walls, a floor and ceiling lights.','supported','sufficient'),
 ('10315129403','In the opening seconds, do the two people move toward or away from the camera?','Away.','During the opening seconds their backs face the camera as they move farther down the corridor.','supported','sufficient'),
 ('2591888090','What color is the stroller canopy?','Blue.','The stroller canopy is bright blue.','contradicted','insufficient'),
 ('2591888090','What color is the shirt of the small child at the handles?','Bright red.','The small child at the stroller handles wears a bright red shirt.','contradicted','insufficient'),
 ('2591888090','Where is the stroller moving?','Through water in a swimming pool.','The stroller wheels are submerged in a swimming pool.','contradicted','insufficient'),
 ('10008553263','What does the performer hold near the mouth?','A large book.','The foreground performer holds a large open book against the mouth.','contradicted','insufficient'),
 ('10008553263','What color is the cloth over the performer\'s shoulders?','Entirely green.','The shoulder cloth is entirely green, without red.','contradicted','insufficient'),
 ('10315129403','Where are the two foreground people?','In an open grassy field.','The two foreground people stand in an open grassy field under the sky.','contradicted','insufficient'),
 ('10315129403','In the opening seconds, do the two people move toward or away from the camera?','Toward.','In the opening seconds, both people face the camera and approach it.','contradicted','insufficient'),
 # True observation, irrelevant support: essential separate sufficiency check.
 ('2591888090','What color is the stroller canopy?','Blue.','Grass is visible beside the sidewalk.','supported','insufficient'),
 ('10008553263','What exact words are being spoken?','Hello everyone.','The performer audibly says exactly "Hello everyone".','unverifiable','unverifiable'),
 ('2591888090','What is the child\'s private reason for pushing the stroller?','To win a prize.','The child privately intends to win a prize by pushing the stroller.','unverifiable','unverifiable'),
 ('10315129403','What will the two people do one minute after this clip ends?','Sit in a cafe.','One minute after the provided clip, the two people sit in a cafe.','unverifiable','unverifiable'),
 ('10008553263','How many times exactly does the performer blink between two sampled frames?','Three.','Between 0.000 and 0.501 seconds the performer blinks exactly three times.','unverifiable','unverifiable')]
assert len(rows)==20
def main():
    dest=P/'calibration_candidates.json';assert not dest.exists()
    panels=json.loads((P/'profile.json').read_text(encoding='utf-8'));by={q['source_id']:q['id'] for q in panels}
    candidates=[]
    for i,(source,q,a,claim,verdict,suff) in enumerate(rows):
        sheet=R/f'runs/evidence-profile-220065/contact-{source}.jpg';assert sheet.exists()
        candidates.append(dict(calibration_id=f'cal-{i:02d}',material_id=by[source],source_id=source,question=q,proposed_answer=a,
            evidence=claim,support='The stated observation is offered as support for the proposed answer.',
            expected_fact_verdict=verdict,expected_sufficiency=suff,review_level='single assistant contact-sheet review',
            contact_sheet_sha256=hashlib.sha256(sheet.read_bytes()).hexdigest(),
            scope='calibration claim, not original benchmark question or gold label'))
    dest.write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(materials=len(candidates),sources=3,sufficient=8,insufficient=8,unverifiable=4,
        factual_supported=9,factual_contradicted=7,factual_unverifiable=4,independent_human_review=False)))
if __name__=='__main__':main()
