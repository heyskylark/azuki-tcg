"""Print decoded continuation traces: python traces.py FAMILY CASE_ID MODEL [MODE] [SAMPLE_SEED]."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
NAMES = {0: 'PASS', 1: 'PLAY->garden', 2: 'PLAY->alley', 6: 'ATTACK', 7: 'EQUIP', 8: 'SPELL', 9: 'DEFEND', 10: 'PORTAL',
         11: 'ACTIVATE', 12: 'ALLEY_ABILITY', 13: 'COST_TARGET', 14: 'TARGET', 16: 'CONFIRM', 18: 'PICK', 19: 'BOTTOM', 20: 'BOTTOM_ALL',
         21: 'SEL->alley', 22: 'SEL->equip', 23: 'SEL->garden', 24: 'TOP'}


def decode(actions, seat, checkpoint):
    hand = list(checkpoint['hand'])
    garden = {int(s.split('@')[1].split(':')[0]): s.split('@')[0] for s in checkpoint['my_garden']}
    alley = {int(s.split('@')[1].split(':')[0]): s.split('@')[0] for s in checkpoint['my_alley']}
    out = []
    for row in actions:
        a = row['action']
        who = 'me' if row['seat'] == seat else 'opp'
        t = a[0]
        if who == 'opp':
            out.append(f"opp:{NAMES.get(t, t)}" + (f"(g{a[1]}->{'LEADER' if a[2] == 5 else 'g%d' % a[2]})" if t == 6 else ''))
            continue
        if t in (1, 2):
            code = hand.pop(a[1]) if a[1] < len(hand) else f'h{a[1]}'
            (garden if t == 1 else alley)[a[2]] = code
            out.append(f"{NAMES[t]}({code}@{a[2]})")
        elif t in (7, 8):
            code = hand.pop(a[1]) if a[1] < len(hand) else f'h{a[1]}'
            out.append(f"{NAMES[t]}({code}" + (f"->{'LEADER' if a[2] == 5 else garden.get(a[2], a[2])})" if t == 7 else ')'))
        elif t == 6:
            out.append(f"ATTACK({'LEADER' if a[1] == 5 else garden.get(a[1], a[1])}->{'FACE' if a[2] == 5 else 'opp g%d' % a[2]})")
        elif t == 10:
            code = alley.pop(a[1], f'a{a[1]}')
            garden[a[2]] = code
            out.append(f"PORTAL({code}->g{a[2]})")
        elif t == 11:
            out.append('LEADER_ABILITY' if a[1] == 5 else f"ABILITY(g{a[1]})")
        elif t == 14:
            out.append(f"TARGET({a[1]})")
        else:
            out.append(f"{NAMES.get(t, t)}{a[1:3]}")
    return out


def main():
    fam, case_id, model = sys.argv[1:4]
    mode = sys.argv[4] if len(sys.argv) > 4 else 'argmax'
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    case = next(c for c in json.loads((OUT / fam / 'cases.json').read_text()) if c['id'] == case_id)
    rows = json.loads((OUT / fam / f'{model}_results.json').read_text())['results']
    row = next(r for r in rows if r['case'] == case_id and r['mode'] == mode and r['sample_seed'] == seed and not r.get('forced_host'))
    print(case_id, model, mode, seed, 'success' if row['success'] else 'FAIL', '| board:', case['checkpoint']['my_garden'], 'alley', case['checkpoint']['my_alley'],
          'opp_hp', case['checkpoint']['opp_hp'], 'my_hp', case['checkpoint']['my_hp'], 'ikz', case['checkpoint']['ikz'])
    print('  ' + ' > '.join(decode(row['actions'], case['seat'], case['checkpoint'])))


if __name__ == '__main__':
    main()
