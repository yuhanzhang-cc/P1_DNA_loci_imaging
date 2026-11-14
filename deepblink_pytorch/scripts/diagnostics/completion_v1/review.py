"""Single-reviewer owner summary: presence != assignment TP; no fake agreement."""
from pathlib import Path
from collections import Counter
from diagnostics.handoff_v2.common import read,table,write,csv_write,fresh,sha
from diagnostics.handoff_v2.reviewer import validate_answers


def summarize(responses,key,packet,output):
    answers=table(responses);sources=table(key);binding=read(packet)
    validate_answers(answers,[r['anonymous_id'] for r in sources],complete=True,expected_packet_id=binding['packet_id'])
    source={r['anonymous_id']:r for r in sources};rows=[]
    counts=Counter((source[r['anonymous_id']]['stratum'],source[r['anonymous_id']].get('auto_geometry_class','not_applicable'),r['candidate_is_real_spot'],r['review_confidence']) for r in answers)
    for (stratum,auto,presence,confidence),n in sorted(counts.items()):rows.append(dict(stratum=stratum,automatic_geometry_proxy=auto,candidate_is_real_spot=presence,confidence=confidence,n=n))
    output=fresh(output);csv_write(output/'presence_geometry_crosstab.private.csv',rows,list(rows[0]))
    write(output/'review_summary.json',dict(cases=len(answers),reviewers=1,presence_counts=dict(Counter(r['candidate_is_real_spot'] for r in answers)),
        responses_sha256=sha(responses),source_key_sha256=sha(key),packet_id=binding['packet_id'],real_human_review_completed=True,
        inter_rater_agreement=None,inter_rater_reason='one reviewer; no invented agreement',
        limitations='enriched diagnostic sample, not population FP rate; spot presence is not assignment TP/FP; geometry proxies remain proxies'))
    return dict(output=str(output),cases=len(answers),human_answers_changed=False)
