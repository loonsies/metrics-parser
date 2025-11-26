from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser import FFXIBattleAnalyzer

INPUT_DIR = Path('input')
files = list(INPUT_DIR.iterdir())

# pick Database and Battle Log
db = next((f for f in files if 'Database' in f.name), None)
battle = next((f for f in files if 'Battle Log' in f.name), None)
print('DB:', db)
print('Battle:', battle)

an = FFXIBattleAnalyzer(str(battle), str(db))
print('Detected boss:', an.boss)
print('Players:', an.players)
