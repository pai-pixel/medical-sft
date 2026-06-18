"""把 audit_1000.jsonl (chosen/reject tail 各 300 char) 压缩成
    chosen/reject tail 各 100 char 的紧凑文本视图, 方便分批 Read 审."""
import json
import os

SRC = os.path.join(os.path.dirname(__file__), 'audit_1000.jsonl')
DST = os.path.join(os.path.dirname(__file__), 'audit_1000_compact.txt')

with open(SRC, encoding='utf-8') as fin, open(DST, 'w', encoding='utf-8') as fout:
    for line in fin:
        r = json.loads(line)
        ct = r['ct'][-110:].replace('\n', ' ')
        rt = r['rt'][-110:].replace('\n', ' ')
        ratio = r['cl'] / max(r['rl'], 1)
        fout.write(
            f"[{r['n']:04d}] id={r['id']:>5} {r['dom']:7} {r['src']:6} "
            f"C={r['cl']:>4} R={r['rl']:>4} r={ratio:.2f}\n"
            f"  P: {r['p'][:80]}\n"
            f"  C: ...{ct}\n"
            f"  R: ...{rt}\n"
        )
print(f'wrote {DST}')
print(f'total lines: {sum(1 for _ in open(DST, encoding="utf-8"))}')
