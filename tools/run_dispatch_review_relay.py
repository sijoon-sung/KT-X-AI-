"""Send the prepared, allowlisted work-order facts to the approved hosted LLM."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from antigravity_transport import generate

parser=argparse.ArgumentParser()
for name in ('request','out','relay-workspace'):
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--agy',default='agy')
args=parser.parse_args()
try:
    request=json.loads(args.request.read_text(encoding='utf-8'))
    candidate,metadata=generate(request,args.relay_workspace,args.agy)
    report=dict(candidate=candidate,transport=metadata)
except (OSError,ValueError,RuntimeError) as exc:
    report=dict(candidate=None,transport=dict(error=type(exc).__name__))
args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
