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

# replicate detection steps from the class for debugging
a = an.battle_log
# battle candidate
mob_mask = a['Flag'].str.contains('Mob', na=False)
mob_rows = a[mob_mask]
print('\nBattle mob rows count:', len(mob_rows))
print('Battle top players in mob rows:')
print(mob_rows['Player Name'].value_counts().head())

# db candidate
b = an.basic
print('\nUnique players from DB (Player column):')
print(sorted(b['Player'].dropna().unique()))
targets = b['Target'].dropna()
print('\nTop DB targets:')
print(targets.value_counts().head())

# filter using DB players
player_set = set(b['Player'].dropna().unique())
filtered = targets[~targets.isin(player_set) & (targets != '') & (targets != '!All Mobs') & (targets != 'System')]
print('\nFiltered DB targets:')
print(filtered.value_counts().head())

print('\nAnalyzer thinks boss:', an.boss)
print('Analyzer players:', an.players)
