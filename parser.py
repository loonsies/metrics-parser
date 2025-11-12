"""
FFXI Battle Log Analyzer - Standalone Parser
Generates comprehensive battle summary images (750x300 PNG) with formatted statistics.

Requires: pandas, svgwrite, playwright, Pillow
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import math
import svgwrite
import re
import base64
import io
import requests
from PIL import Image, ImageFilter


# ============================================================================
# FFXI Battle Analyzer Class
# ============================================================================

class FFXIBattleAnalyzer:
    def __init__(self, battle_log_path, basic_path, catalog_path=None, background_image=None):
        """Initialize the battle analyzer with CSV file paths"""
        self.battle_log = pd.read_csv(battle_log_path)
        self.basic = pd.read_csv(basic_path)
        self.catalog = pd.read_csv(catalog_path) if catalog_path else None
        self.background_image = background_image
        
        # Parse time column
        self.battle_log['Time'] = pd.to_datetime(self.battle_log['Time'], format='%H:%M:%S')
        
        # Get unique players and boss
        self.players = sorted(self.battle_log['Player Name'].unique())
        if 'NONE' in self.players:
            self.players.remove('NONE')
        
        # Detect boss (from Mob Death or mob actions)
        self.boss = self._detect_boss()

        # Remove boss from players list if present (boss rows sometimes appear in 'Player Name')
        try:
            if self.boss in self.players:
                self.players.remove(self.boss)
        except Exception:
            pass

        # Extract player job/subjob information (if available)
        self.player_jobs = self._extract_player_jobs()

        # Assign consistent colors to each player
        self.player_colors = self._assign_player_colors()

    def _extract_player_jobs(self):
        """Try to extract job/subjob for each player from available CSVs.

        Heuristics:
        - Look for explicit 'Job' or 'SubJob' columns in `basic`, `catalog`, or `battle_log`.
        - Search `battle_log` text fields (Note, Action) for known job abbreviations.
        Returns mapping: player_name -> "JOB[/SUB]" or empty string.
        """
        # Build a normalized lookup (lowercased, stripped) so we can match names across CSVs
        def norm(n):
            try:
                return str(n).strip().lower()
            except Exception:
                return ''

        # initialize temp mapping for all known players
        temp = {norm(p): '' for p in self.players}

        dfs = [self.basic, self.catalog, self.battle_log]
        name_cols = ['Player Name', 'Actor', 'Name']
        # column name alternatives for Job/SubJob
        job_cols = ['Job', 'JOB']
        subjob_cols = ['SubJob', 'Sub Job', 'Subjob', 'Sub-Job', 'SUBJOB']

        # First, try explicit Job/SubJob columns in any dataframe
        for df in dfs:
            if df is None:
                continue
            cols = set(df.columns)
            # find a name column present in this df
            name_col = None
            for nc in name_cols:
                if nc in cols:
                    name_col = nc
                    break
            if not name_col:
                continue

            # try to detect job/subjob columns flexibly
            found_job_col = next((c for c in cols if c.lower() == 'job'), None)
            found_sub_col = next((c for c in cols if c.replace(' ', '').lower() in {s.replace(' ', '').lower() for s in subjob_cols}), None)

            if found_job_col:
                for _, row in df.iterrows():
                    try:
                        name = row.get(name_col, '')
                        if pd.isna(name) or not name:
                            continue
                        n = norm(name)
                        job = ''
                        sub = ''
                        if not pd.isna(row.get(found_job_col, '')):
                            job = str(row.get(found_job_col, '')).strip()
                        if found_sub_col and not pd.isna(row.get(found_sub_col, '')):
                            sub = str(row.get(found_sub_col, '')).strip()
                        if job and sub:
                            if not temp.get(n):
                                temp[n] = f"{job}/{sub}"
                        elif job:
                            if not temp.get(n):
                                temp[n] = job
                    except Exception:
                        continue

        # If still missing, search free text for job and subjob abbreviations in battle_log text
        # Use word-boundary-aware patterns and explicit JOB/SUB patterns so we do not match
        # substrings inside ability names (e.g. 'WARCRY' -> should NOT match 'WAR').
        known_jobs = ['WAR','PLD','DRK','MNK','SAM','NIN','DRG','BRD','RNG','COR','PUP','BLM','RDM','SMN','WHM','SCH','BLU','GEO','DNC','BST','THF']
        # patterns: explicit JOB/SUB or JOB (SUB), or standalone JOB token (with word boundaries)
        job_alts = '|'.join(known_jobs)
        job_sub_pattern = re.compile(rf"\b({job_alts})\s*(?:/|\(|-)\s*([A-Z]{{2,4}})\b")
        single_job_pattern = re.compile(rf"\b({job_alts})\b")

        if self.battle_log is not None:
            for _, row in self.battle_log.iterrows():
                try:
                    pname = row.get('Player Name', '')
                    if pd.isna(pname) or not pname:
                        continue
                    n = norm(pname)
                    if temp.get(n):
                        continue
                    text_fields = []
                    for k in ('Note','Action'):
                        if k in row.index and not pd.isna(row.get(k, '')):
                            text_fields.append(str(row.get(k, '')))
                    combined = ' '.join(text_fields).upper()

                    # First, try explicit JOB/SUB patterns like "WAR/SAM" or "WAR (SAM)"
                    m = job_sub_pattern.search(combined)
                    if m:
                        job = m.group(1)
                        sub = m.group(2) or ''
                        temp[n] = f"{job}/{sub}".rstrip('/')
                        continue

                    # Next, look for a standalone job token with word boundaries.
                    # This avoids matching substrings inside ability names (e.g. WARCRY).
                    m2 = single_job_pattern.search(combined)
                    if m2:
                        temp[n] = m2.group(1)
                except Exception:
                    continue

        # Map back to original player names (preserve original casing/whitespace)
        jobs = {}
        for p in self.players:
            jobs[p] = temp.get(norm(p), '')

        return jobs

    def get_player_label(self, name):
        """Return a display label for a player including job/subjob if available."""
        # Return the player name (no job) — job is displayed separately by renderers
        if not name:
            return ''
        return str(name)

    def get_player_job(self, name):
        """Return the player's job/subjob string (e.g. 'WAR/SAM') or empty string."""
        if not name:
            return ''
        return self.player_jobs.get(name, '') if hasattr(self, 'player_jobs') else ''
    
    def _assign_player_colors(self):
        """Assign a unique consistent color to each player"""
        # Use BASE_COLORS from environment (.env) so colors can be customized
        base_colors = BASE_COLORS
        
        player_colors = {}
        for i, player in enumerate(self.players):
            player_colors[player] = base_colors[i % len(base_colors)]
        
        return player_colors
    
    def _detect_boss(self):
        """Detect the boss name from the battle log"""
        mob_death = self.battle_log[self.battle_log['Flag'] == 'Mob Death']
        if not mob_death.empty:
            return mob_death.iloc[0]['Player Name']
        
        # Try to find from mob actions
        mob_actions = self.battle_log[self.battle_log['Flag'].str.contains('Mob', na=False)]
        if not mob_actions.empty:
            return mob_actions.iloc[0]['Player Name']
        
        return "Unknown Boss"
    
    def calculate_dps_stats(self):
        """Calculate DPS statistics for all players (new format)"""
        # Use new event types for damage
        damage_flags = ['Melee', 'Weaponskill', 'Skillchain', 'Offensive Magic']
        damage_df = self.battle_log[self.battle_log['Flag'].isin(damage_flags)]

        dps_stats = {}
        for player in self.players:
            player_damage = damage_df[damage_df['Player Name'] == player]

            # Convert damage to numeric, handling '---' and other non-numeric values
            damage_values = pd.to_numeric(player_damage['Damage'], errors='coerce').fillna(0)

            total_damage = damage_values.sum()

            # Calculate fight duration
            if not player_damage.empty:
                start_time = player_damage['Time'].min()
                end_time = player_damage['Time'].max()
                duration_seconds = (end_time - start_time).total_seconds()
                if duration_seconds > 0:
                    dps = total_damage / duration_seconds
                else:
                    dps = 0
            else:
                dps = 0
                duration_seconds = 0

            # Get breakdown by damage type
            breakdown = {}
            for flag in damage_flags:
                flag_damage = player_damage[player_damage['Flag'] == flag]
                flag_values = pd.to_numeric(flag_damage['Damage'], errors='coerce').fillna(0)
                breakdown[flag] = flag_values.sum()

            dps_stats[player] = {
                'total_damage': total_damage,
                'dps': dps,
                'duration': duration_seconds,
                'breakdown': breakdown
            }

        return dps_stats
    
    def calculate_healing_stats(self):
        """Calculate healing given by each player from Database CSV (Player = healer)"""
        healing_stats = {p: {'total_healing': 0, 'heal_count': 0, 'avg_heal': 0} for p in self.players}
        try:
            if self.basic is not None:
                df = self.basic
                for p in self.players:
                    rows = df[(df['Player'] == p) & (df['Trackable'].isin(['Spells Healing', 'All Sources Healing'])) & (df['Metric'] == 'Total')]
                    total = rows['Value'].astype(float).sum()
                    count = len(rows)
                    avg = total / count if count > 0 else 0
                    healing_stats[p]['total_healing'] = total
                    healing_stats[p]['heal_count'] = count
                    healing_stats[p]['avg_heal'] = avg
        except Exception:
            pass
        return healing_stats
    
    def calculate_healing_received_stats(self):
        """Calculate healing received by each player from Database CSV (Target = recipient).

        The Database CSV contains both per-player targets and aggregate targets like '!All Mobs'.
        We include aggregate targets in the results so totals reconcile with heal-given.
        """
        # Only include actual players as recipients. Aggregate targets like '!All Mobs'
        # represent group-wide stats (not an individual) and should not appear in
        # the per-player "Heal Received" breakdown.
        healing_received = {p: {'total_received': 0, 'heal_count': 0} for p in self.players}
        try:
            if self.basic is not None:
                df = self.basic
                # Normalize target values and filter to only real players we know about
                targets_mask = df['Target'].isin(self.players)
                rows = df[targets_mask & df['Trackable'].isin(['Spells Healing', 'All Sources Healing']) & (df['Metric'] == 'Total')]
                for p in self.players:
                    pr = rows[rows['Target'] == p]
                    total = pr['Value'].astype(float).sum() if not pr.empty else 0
                    count = len(pr)
                    healing_received[p]['total_received'] = total
                    healing_received[p]['heal_count'] = count
        except Exception:
            pass
        return healing_received
    
    def calculate_damage_taken(self):
        """Calculate damage taken by each player (custom for your Database CSV)"""
        damage_taken_stats = {}

        for player in self.players:
            # Sum 'Total' from 'Defense Damage Taken Total' for each player
            if 'Player' in self.basic.columns:
                player_basic = self.basic[
                    (self.basic['Player'] == player) &
                    (self.basic['Trackable'] == 'Defense Damage Taken Total') &
                    (self.basic['Metric'] == 'Total')
                ]
            else:
                player_basic = pd.DataFrame()

            if not player_basic.empty:
                total_damage_taken = player_basic['Value'].astype(float).sum()
            else:
                total_damage_taken = 0

            # Check for deaths
            deaths = self.battle_log[
                (self.battle_log['Player Name'] == player) &
                (self.battle_log['Flag'] == 'Death')
            ]
            death_count = len(deaths)

            damage_taken_stats[player] = {
                'total_damage_taken': total_damage_taken,
                'deaths': death_count
            }

        return damage_taken_stats
    
    def calculate_weaponskill_stats(self):
        """Calculate detailed weaponskill statistics for all players (new format)"""
        ws_df = self.battle_log[self.battle_log['Flag'] == 'Weaponskill']

        ws_stats = defaultdict(lambda: defaultdict(list))

        for _, row in ws_df.iterrows():
            player = row['Player Name']
            ws_name = row['Action']
            damage = pd.to_numeric(row['Damage'], errors='coerce')

            if pd.notna(damage) and damage > 0:
                ws_stats[player][ws_name].append(damage)

        # Calculate min, max, avg for each WS
        ws_summary = []
        for player in sorted(ws_stats.keys()):
            for ws_name in sorted(ws_stats[player].keys()):
                damages = ws_stats[player][ws_name]
                ws_summary.append({
                    'Player': player,
                    'Weaponskill': ws_name,
                    'Count': len(damages),
                    'Min': min(damages),
                    'Max': max(damages),
                    'Avg': sum(damages) / len(damages),
                    'Total': sum(damages)
                })

        return ws_summary
    
    def get_key_moments(self):
        """Extract key moments from the battle - deaths and victory"""
        key_moments = []
        
        # Deaths
        deaths = self.battle_log[self.battle_log['Flag'] == 'Death']
        for _, row in deaths.iterrows():
            player = row['Player Name']
            death_time = row['Time']
            
            # Killer name is stored in the Note column of the Death row
            killer_name = row.get('Note', '').strip() if 'Note' in row else ''
            
            # Try to find the last weaponskill used by the player before death
            last_ws_info = ''
            try:
                ws_rows = self.battle_log[
                    (self.battle_log['Player Name'] == player) &
                    (self.battle_log['Flag'] == 'WS') &
                    (self.battle_log['Time'] < death_time)
                ]
                if not ws_rows.empty:
                    last_ws = ws_rows.sort_values('Time', ascending=False).iloc[0]
                    ws_name = last_ws.get('Action', '')
                    ws_dmg = None
                    if 'Damage' in last_ws:
                        try:
                            ws_dmg = pd.to_numeric(last_ws.get('Damage'), errors='coerce')
                        except Exception:
                            ws_dmg = None
                    if ws_name:
                        if ws_dmg is not None and not pd.isna(ws_dmg):
                            last_ws_info = f"Last WS: {ws_name} ({int(ws_dmg):,} dmg)"
                        else:
                            last_ws_info = f"Last WS: {ws_name}"
            except Exception:
                last_ws_info = ''

            # Try to find the killer's attack (most recent action by the killer before death)
            killer_attack_info = ''
            if killer_name:
                try:
                    # Find recent actions by the killer
                    killer_actions = self.battle_log[
                        (self.battle_log['Player Name'] == killer_name) &
                        (self.battle_log['Time'] <= death_time) &
                        (self.battle_log['Time'] >= death_time - timedelta(seconds=5))
                    ].sort_values('Time', ascending=False)
                    
                    if not killer_actions.empty:
                        # Get the most recent action (likely the killing blow)
                        killer_action = killer_actions.iloc[0]
                        attack_name = killer_action.get('Action', '')
                        attack_dmg = None
                        if 'Damage' in killer_action:
                            try:
                                attack_dmg = pd.to_numeric(killer_action.get('Damage'), errors='coerce')
                                if pd.notna(attack_dmg) and attack_dmg > 0:
                                    attack_dmg = int(attack_dmg)
                                else:
                                    attack_dmg = None
                            except Exception:
                                attack_dmg = None
                        
                        if attack_name and attack_dmg:
                            killer_attack_info = f"Killed by {killer_name}'s {attack_name} ({attack_dmg:,} dmg)"
                        elif attack_name:
                            killer_attack_info = f"Killed by {killer_name}'s {attack_name}"
                        elif attack_dmg:
                            killer_attack_info = f"Killed by {killer_name} ({attack_dmg:,} dmg)"
                        else:
                            killer_attack_info = f"Killed by {killer_name}"
                    else:
                        killer_attack_info = f"Killed by {killer_name}"
                except Exception:
                    killer_attack_info = f"Killed by {killer_name}" if killer_name else ''

            # Build the action text
            parts = []
            if killer_attack_info:
                parts.append(killer_attack_info)
            if last_ws_info:
                parts.append(last_ws_info)
            
            action_text = '; '.join(parts) if parts else 'Died'

            key_moments.append({
                'time': row['Time'].strftime('%H:%M:%S'),
                'player': player,
                'action': action_text,
                'type': 'Death'
            })
        
        # Boss death
        boss_death = self.battle_log[self.battle_log['Flag'] == 'Mob Death']
        for _, row in boss_death.iterrows():
            key_moments.append({
                'time': row['Time'].strftime('%H:%M:%S'),
                'player': row['Player Name'],
                'action': 'Boss Defeated!',
                'type': 'Victory'
            })
        
        # Sort by time
        key_moments.sort(key=lambda x: x['time'], reverse=True)
        
        return key_moments
    
    def get_fight_duration(self):
        """Calculate fight duration from first damage event to boss death (or last damage event).

        This uses the same damage flags as other calculations. If no damage events are
        present, falls back to the overall min/max times.
        Returns (human_readable, seconds)
        """
        damage_flags = ['Melee', 'WS', 'SC', 'Nuking', 'Spells']
        try:
            # find the first damage event (exclude boss self-damage)
            dmg_events = self.battle_log[
                (self.battle_log['Flag'].isin(damage_flags)) &
                (self.battle_log['Player Name'] != self.boss)
            ]
            if not dmg_events.empty:
                start_time = dmg_events['Time'].min()
            else:
                start_time = self.battle_log['Time'].min()

            # prefer explicit boss death time if present
            mob_death_rows = self.battle_log[self.battle_log['Flag'] == 'Mob Death']
            if not mob_death_rows.empty:
                md = mob_death_rows[mob_death_rows['Player Name'] == self.boss]
                if not md.empty:
                    end_time = md['Time'].iloc[0]
                else:
                    # no mob death for this boss name; fall back to last damage event
                    if not dmg_events.empty:
                        end_time = dmg_events['Time'].max()
                    else:
                        end_time = self.battle_log['Time'].max()
            else:
                # no mob death rows at all -> use last damage or overall max
                if not dmg_events.empty:
                    end_time = dmg_events['Time'].max()
                else:
                    end_time = self.battle_log['Time'].max()

            duration = (end_time - start_time).total_seconds()
            if duration < 0:
                duration = 0
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}m {seconds}s", duration
        except Exception:
            # fallback to original behavior
            start_time = self.battle_log['Time'].min()
            end_time = self.battle_log['Time'].max()
            duration = (end_time - start_time).total_seconds()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}m {seconds}s", duration

    def calculate_boss_hp_milestones(self):
        """Calculate when boss HP reached 75%, 50%, 25% remaining during the fight.
        
        The boss starts at 100% HP and we deal damage until it reaches 0%.
        We infer the total boss HP from the total damage dealt.
        
        Milestones represent boss HP REMAINING:
        - 75% milestone = when boss HP drops to 75% (we've dealt 25% of boss HP in damage)
        - 50% milestone = when boss HP drops to 50% (we've dealt 50% of boss HP in damage)
        - 25% milestone = when boss HP drops to 25% (we've dealt 75% of boss HP in damage)
        """
        damage_flags = ['Melee', 'WS', 'SC', 'Nuking', 'Spells']
        boss_name = self.boss
        
        try:
            # Fight start is the first damage event against the boss
            first_dmg = self.battle_log[
                (self.battle_log['Flag'].isin(damage_flags)) &
                (self.battle_log['Player Name'] != boss_name)
            ]
            if first_dmg.empty:
                return {}
            start_time = first_dmg['Time'].min()
        except Exception:
            return {}

        # Find boss death time
        mob_death_rows = self.battle_log[self.battle_log['Flag'] == 'Mob Death']
        end_time = self.battle_log['Time'].max()
        if not mob_death_rows.empty:
            md = mob_death_rows[mob_death_rows['Player Name'] == boss_name]
            if not md.empty:
                end_time = md['Time'].iloc[0]

        # Get all damage events against the boss
        dmg_events = self.battle_log[
            (self.battle_log['Flag'].isin(damage_flags)) &
            (self.battle_log['Player Name'] != boss_name) &
            (self.battle_log['Time'] >= start_time) &
            (self.battle_log['Time'] <= end_time)
        ].copy()

        if dmg_events.empty:
            return {'start_time': start_time.strftime('%H:%M:%S')}

        dmg_events['Damage_val'] = pd.to_numeric(dmg_events['Damage'], errors='coerce').fillna(0)
        dmg_events = dmg_events.sort_values('Time')
        dmg_events['cum'] = dmg_events['Damage_val'].cumsum()
        
        total_damage = float(dmg_events['cum'].iloc[-1])
        if total_damage <= 0:
            return {'start_time': start_time.strftime('%H:%M:%S')}

        milestones = {}
        milestones['start_time'] = start_time.strftime('%H:%M:%S')
        
        # Milestones: when cumulative damage crosses these percentages of total
        # 25% damage dealt = boss at 75% HP
        # 50% damage dealt = boss at 50% HP
        # 75% damage dealt = boss at 25% HP
        for label, damage_pct in [('75%', 0.25), ('50%', 0.50), ('25%', 0.75)]:
            thresh = total_damage * damage_pct
            cross = dmg_events[dmg_events['cum'] >= thresh]
            if not cross.empty:
                milestones[label] = cross.iloc[0]['Time'].strftime('%H:%M:%S')

        milestones['inferred_total_damage'] = int(total_damage)
        return milestones


# ============================================================================
# SVG Builder Functions & Utilities
# ============================================================================

# Canvas and IO defaults (can be overridden via .env)
W = int(os.getenv('CANVAS_WIDTH', '750'))
H = int(os.getenv('CANVAS_HEIGHT', '300'))
INPUT_DIR = Path(os.getenv('INPUT_DIR', 'input'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'output'))
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FONT_FAMILY = os.getenv('FONT_FAMILY', 'Roboto, Inter, Arial, sans-serif')

# Color and appearance defaults (override in .env as needed)
# BASE_COLORS should be a comma-separated list of hex values
_base_colors_env = os.getenv('BASE_COLORS', '#ff6b6b,#4ecdc4,#45b7d1,#f9ca24,#6c5ce7,#00d2d3,#fd79a8,#fdcb6e,#a29bfe,#55efc4,#ff7675,#74b9ff')
BASE_COLORS = [c.strip() for c in _base_colors_env.split(',') if c.strip()]

BG_OVERLAY_COLOR = os.getenv('BG_OVERLAY_COLOR', '#060606')
BG_OVERLAY_OPACITY = float(os.getenv('BG_OVERLAY_OPACITY', '0.18'))
TITLE_COLOR = os.getenv('TITLE_COLOR', '#ffffff')
SUBTITLE_COLOR = os.getenv('SUBTITLE_COLOR', '#cccccc')
PANEL_BG_COLOR = os.getenv('PANEL_BG_COLOR', '#1b1b1b')
TEXT_PRIMARY = os.getenv('TEXT_PRIMARY', '#ffffff')
TEXT_SECONDARY = os.getenv('TEXT_SECONDARY', '#cccccc')
STROKE_COLOR = os.getenv('STROKE_COLOR', '#262626')
GRID_LINE_COLOR = os.getenv('GRID_LINE_COLOR', '#2a2a2a')
DEFAULT_PLAYER_COLOR = os.getenv('DEFAULT_PLAYER_COLOR', '#888')
FIGHT_START_COLOR = os.getenv('FIGHT_START_COLOR', '#4ecdc4')

# Image processing defaults
IMG_BLUR_RADIUS = float(os.getenv('IMG_BLUR_RADIUS', '4'))
IMG_DARKEN_ALPHA = float(os.getenv('IMG_DARKEN_ALPHA', '0.28'))

# --------------------------------------------------------------------------
# Lightweight .env loader (avoids external dependency on python-dotenv)
# Loads key=value pairs into os.environ if not already defined.
# --------------------------------------------------------------------------
def _load_env(path: Path = Path('.env')):
    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as e:
        print(f"Warning: failed to load .env file: {e}")

_load_env()

# Read webhook from env (leave blank if not set)
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')


def _approx_text_width(text, font_size=11):
    """Rough approximation: average char width ~= 0.6 * font_size"""
    return len(str(text)) * (font_size * 0.6)


def _embed_font_css(dwg):
    """If Coolvetica.ttf exists in cwd or fonts/, embed it as a base64 @font-face so Playwright will render it."""
    candidates = [Path('Coolvetica.ttf'), Path('fonts') / 'Coolvetica.ttf']
    for p in candidates:
        if p.exists():
            try:
                font_bytes = p.read_bytes()
                b64 = base64.b64encode(font_bytes).decode('ascii')
                css = f"@font-face {{ font-family: 'Coolvetica'; src: url('data:font/truetype;base64,{b64}') format('truetype'); font-weight: normal; font-style: normal; }}"
                dwg.defs.add(dwg.style(css))
                return True
            except Exception:
                return False
    return False


def _prepare_bg_data_uri(path, w, h, blur_radius=4, darken_alpha=0.28):
    """Load image, produce a cover-resized, blurred and darkened PNG data URI for embedding in SVG.
    This avoids SVG stretching and ensures the background fully covers the 750x300 area.
    """
    try:
        im = Image.open(path)
        if im.mode != 'RGBA':
            im = im.convert('RGBA')

        src_w, src_h = im.size
        # compute scale to cover
        scale = max(w / src_w, h / src_h)
        new_size = (int(src_w * scale + 0.5), int(src_h * scale + 0.5))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        # center-crop to target
        left = (im.width - w) // 2
        top = (im.height - h) // 2
        im = im.crop((left, top, left + w, top + h))

        # blur & darken overlay
        im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        overlay = Image.new('RGBA', (w, h), (6, 6, 6, int(255 * darken_alpha)))
        im = Image.alpha_composite(im, overlay)

        bio = io.BytesIO()
        im.convert('RGB').save(bio, format='PNG')
        b64 = base64.b64encode(bio.getvalue()).decode('ascii')
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def create_base(title, subtitle, analyzer, width=None, height=None):
    """Common base: background, title.

    width/height default to global W/H but can be overridden to create taller images.
    """
    local_w = int(width) if width is not None else W
    local_h = int(height) if height is not None else H

    dwg = svgwrite.Drawing(size=(f'{local_w}px', f'{local_h}px'))
    try:
        dwg.attribs['font-family'] = FONT_FAMILY
    except Exception:
        pass

    _embed_font_css(dwg)

    # background image if available -> embed as data URI after processing to cover area
    if analyzer.background_image and Path(analyzer.background_image).exists():
        bg_data = _prepare_bg_data_uri(analyzer.background_image, local_w, local_h, blur_radius=IMG_BLUR_RADIUS, darken_alpha=IMG_DARKEN_ALPHA)
        if bg_data:
            dwg.add(dwg.image(href=bg_data, insert=(0, 0), size=(local_w, local_h)))
    # overlay (subtle) - color/opac configurable via .env
    dwg.add(dwg.rect(insert=(0, 0), size=(local_w, local_h), fill=BG_OVERLAY_COLOR, fill_opacity=BG_OVERLAY_OPACITY))
    # title
    dwg.add(dwg.text(title, insert=(local_w / 2, 26), text_anchor='middle', fill=TITLE_COLOR, font_size=16, font_weight='bold'))
    dwg.add(dwg.text(subtitle, insert=(local_w / 2, 44), text_anchor='middle', fill=SUBTITLE_COLOR, font_size=11))
    return dwg


def save_svg_and_png(dwg, name_base, width=None, height=None):
    """Save PNG only (skip SVG) and render via Playwright.

    width/height default to globals if not provided.
    """
    png_path = OUTPUT_DIR / f"{name_base}.png"
    svg_bytes = dwg.tostring().encode('utf-8')

    # Render via Playwright sync (no SVG file saved)
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f'Playwright not available: {e}')
        return None

    # Use temp SVG in memory
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False, encoding='utf-8') as tmp:
        tmp.write(dwg.tostring())
        tmp_path = Path(tmp.name)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            vw = int(width) if width is not None else W
            vh = int(height) if height is not None else H
            context = browser.new_context(viewport={'width': vw, 'height': vh}, device_scale_factor=1)
            page = context.new_page()
            page.goto(tmp_path.resolve().as_uri())
            page.wait_for_timeout(200)
            page.screenshot(path=str(png_path), full_page=False)
            browser.close()
        print('Wrote', png_path)
        return png_path
    finally:
        tmp_path.unlink(missing_ok=True)


def upload_to_discord(png_path, webhook_url):
    """Upload a PNG image to Discord via webhook"""
    try:
        with open(png_path, 'rb') as f:
            files = {'file': (png_path.name, f, 'image/png')}
            response = requests.post(webhook_url, files=files)
            if response.status_code in (200, 204):
                print(f'  → Uploaded {png_path.name} to Discord')
                return True
            else:
                print(f'  → Discord upload failed for {png_path.name}: {response.status_code} - {response.text}')
                return False
    except Exception as e:
        print(f'  → Error uploading {png_path.name}: {e}')
        return False


def send_discord_message(content, webhook_url):
    """Send a simple text message to Discord via webhook."""
    try:
        response = requests.post(webhook_url, json={'content': content})
        if response.status_code in (200, 204):
            print('  → Sent Discord message')
            return True
        else:
            print(f'  → Discord message failed: {response.status_code} - {response.text}')
            return False
    except Exception as e:
        print(f'  → Error sending Discord message: {e}')
        return False


def build_dps(analyzer, suffix=''):
    """DPS: bars + pie"""
    dps_stats = analyzer.calculate_dps_stats()
    players = [p for p,s in dps_stats.items() if s['total_damage']>0]
    players = sorted(players, key=lambda p: dps_stats[p]['total_damage'], reverse=True)
    dwg = create_base(f"DPS Summary - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer)
    left_x, left_y, left_w, left_h = 12, 56, 360, 228
    dwg.add(dwg.rect(insert=(left_x,left_y), size=(left_w,left_h), rx=8, ry=8, fill='none', stroke='none'))
    if players:
        max_dps = max(dps_stats[p]['dps'] for p in players)
        # Use full width: 15% for names, small gap, rest for bars
        min_margin = 24  # minimum space from panel edge
        available_w = left_w - (2 * min_margin)
        gap_between = 8
        name_w = available_w * 0.15
        bar_total_w = available_w * 0.85
        total_block_w = name_w + gap_between + bar_total_w
        block_start_x = left_x + min_margin
        
        num = len(players)
        gap = 8
        bar_h = max(14, (left_h - (num+1)*gap)//num)
        cy = left_y + gap
        for p in players:
            color = analyzer.player_colors.get(p, DEFAULT_PLAYER_COLOR)
            dps = dps_stats[p]['dps']
            tot = dps_stats[p]['total_damage']
            bw = int((dps / max_dps) * bar_total_w) if max_dps>0 else 0
            name_x = block_start_x + name_w - 4
            name_text = analyzer.get_player_label(p)
            job_text = analyzer.get_player_job(p)
            # Draw name (primary) and job (smaller, secondary) stacked within the bar name area
            name_y = cy + (bar_h / 2) - 4
            job_y = cy + (bar_h / 2) + 6
            dwg.add(dwg.text(name_text, insert=(name_x, name_y), fill=color, font_size=12, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
            if job_text:
                dwg.add(dwg.text(job_text, insert=(name_x, job_y), fill=TEXT_SECONDARY, font_size=9, text_anchor='end', **{'dominant-baseline':'middle'}))
            bar_x = block_start_x + name_w + gap_between
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bar_total_w, bar_h-2), fill=PANEL_BG_COLOR, rx=2, ry=2))
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bw, bar_h-2), fill=color, rx=2, ry=2))
            label_text = f"{dps:.1f} DPS ({int(tot):,})"
            lx = bar_x + bar_total_w - 6
            dwg.add(dwg.text(label_text, insert=(lx, cy + bar_h/2), text_anchor='end', fill=TEXT_PRIMARY, font_size=11, **{'dominant-baseline':'middle'}))
            cy += bar_h + gap
    # pie
    right_margin = 12
    right_x0 = left_x + left_w + right_margin
    right_w = W - right_x0 - right_margin
    r = min(92, int(min(right_w, left_h) * 0.45))
    legend_items = len(players)
    legend_cols = max(1, int(right_w // 120))
    legend_rows = math.ceil(max(0, legend_items) / legend_cols) if legend_items > 0 else 0
    legend_h = legend_rows * 18
    pie_group_h = 2 * r + 8 + legend_h
    top_y = left_y + (left_h - pie_group_h) / 2
    cx = right_x0 + right_w/2
    cy = top_y + r
    total = sum(dps_stats[p]['total_damage'] for p in players)
    a0 = -math.pi/2
    for p in players:
        dmg = dps_stats[p]['total_damage']
        frac = dmg/total if total>0 else 0
        a1 = a0 + frac*2*math.pi
        x1 = cx + r*math.cos(a0); y1 = cy + r*math.sin(a0)
        x2 = cx + r*math.cos(a1); y2 = cy + r*math.sin(a1)
        large = 1 if (a1-a0)>math.pi else 0
        d = f"M {cx},{cy} L {x1},{y1} A {r},{r} 0 {large},1 {x2},{y2} Z"
        dwg.add(dwg.path(d=d, fill=analyzer.player_colors.get(p, DEFAULT_PLAYER_COLOR), stroke=STROKE_COLOR, stroke_width=0.6))
        mid = (a0 + a1) / 2
        pct = frac * 100
        if pct >= 3:
            tx = cx + (r * 0.55) * math.cos(mid)
            ty = cy + (r * 0.55) * math.sin(mid)
            dwg.add(dwg.text(f"{pct:.0f}%", insert=(tx, ty), text_anchor='middle', fill=TEXT_PRIMARY, font_size=11, **{'dominant-baseline':'middle'}))
        a0 = a1
    legend_block_w = min(right_w, max(120, legend_cols * 120))
    lx_start = cx - legend_block_w / 2
    lx = lx_start
    baseline_y = top_y + 2 * r + 12
    col_count = 0
    cell_w = legend_block_w / max(1, legend_cols)
    for idx, p in enumerate(players):
        color = analyzer.player_colors.get(p, DEFAULT_PLAYER_COLOR)
        cell_center_x = lx + cell_w/2
        swatch_x = cell_center_x - 30
        swatch_y = baseline_y - 6
        text_x = cell_center_x - 6
        dwg.add(dwg.rect(insert=(swatch_x, swatch_y), size=(12,12), fill=color))
        player_label = analyzer.get_player_label(p)
        dwg.add(dwg.text(player_label, insert=(text_x, baseline_y), fill=color, font_size=11, **{'dominant-baseline':'middle'}))
        lx += cell_w
        col_count += 1
        if col_count >= legend_cols:
            col_count = 0
            lx = lx_start
            baseline_y += 18
    return save_svg_and_png(dwg, f'dps_summary{suffix}')


def build_healing(analyzer, suffix=''):
    """Healing: split left (healers) and right (recipients)"""
    healing = analyzer.calculate_healing_stats()
    healing_received = analyzer.calculate_healing_received_stats()
    healers = {k:v for k,v in healing.items() if v['total_healing']>0}
    players = sorted(healers.keys(), key=lambda p: healers[p]['total_healing'], reverse=True)
    dwg = create_base(f"Healing Summary - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer)
    
    left_x, left_y, left_w, left_h = 12, 56, 360, 228
    dwg.add(dwg.rect(insert=(left_x,left_y), size=(left_w,left_h), rx=6, ry=6, fill='none', stroke='none'))
    
    title_reserved = 28
    if players:
        max_h = max(healers[p]['total_healing'] for p in players)
        # Use full width: 15% for names, small gap, rest for bars
        min_margin = 24  # minimum space from panel edge
        available_w = left_w - (2 * min_margin)
        gap_between = 8
        name_w = available_w * 0.15
        bar_total_w = available_w * 0.85
        total_block_w = name_w + gap_between + bar_total_w
        block_start_x = left_x + min_margin
        
        gap=10
        num=len(players)
        data_h = left_h - title_reserved
        bar_h = max(16, (data_h - (num+1)*gap)//num)
        name_font_size = 14 if num <= 4 else 13
        value_font_size = 12 if num <= 4 else 11
        cy = left_y + title_reserved + gap
        for p in players:
            color = analyzer.player_colors.get(p, DEFAULT_PLAYER_COLOR)
            val = healers[p]['total_healing']
            cnt = healers[p]['heal_count']
            bw = int((val/max_h)*bar_total_w) if max_h>0 else 0
            name_x = block_start_x + name_w - 4
            label = analyzer.get_player_label(p)
            dwg.add(dwg.text(label, insert=(name_x, cy + bar_h/2), fill=color, font_size=name_font_size, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
            bar_x = block_start_x + name_w + gap_between
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bar_total_w, bar_h-3), fill=PANEL_BG_COLOR, rx=2, ry=2))
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bw, bar_h-3), fill=color, rx=2, ry=2))
            label_text = f"{val:,.0f} HP ({cnt})"
            lx = bar_x + bar_total_w - 6
            dwg.add(dwg.text(label_text, insert=(lx, cy + bar_h/2), text_anchor='end', fill=TEXT_PRIMARY, font_size=value_font_size, **{'dominant-baseline':'middle'}))
            cy += bar_h + gap
    
    right_x, right_y, right_w, right_h = left_x + left_w + 12, 56, 342, 228
    dwg.add(dwg.rect(insert=(right_x,right_y), size=(right_w,right_h), rx=6, ry=6, fill='none', stroke='none'))
    recipients = {k:v for k,v in healing_received.items() if v['total_received']>0}
    if recipients:
        recipients_sorted = sorted(recipients.keys(), key=lambda p: recipients[p]['total_received'], reverse=True)
        max_r = max(recipients[p]['total_received'] for p in recipients_sorted)
        # Use full width: 15% for names, small gap, rest for bars
        min_margin_r = 24  # minimum space from panel edge
        available_w_r = right_w - (2 * min_margin_r)
        gap_between_r = 8
        name_w_r = available_w_r * 0.15
        bar_total_w_r = available_w_r * 0.85
        total_block_w_r = name_w_r + gap_between_r + bar_total_w_r
        block_start_x_r = right_x + min_margin_r
        
        num_r = len(recipients_sorted)
        gap_r = 10
        data_h_r = right_h - title_reserved
        bar_h_r = max(16, (data_h_r - (num_r+1)*gap_r)//num_r)
        name_font_r = 14 if num_r <= 4 else 13
        value_font_r = 12 if num_r <= 4 else 11
        cy_r = right_y + title_reserved + gap_r
        for p in recipients_sorted:
            color = analyzer.player_colors.get(p, DEFAULT_PLAYER_COLOR)
            val_r = recipients[p]['total_received']
            cnt_r = recipients[p]['heal_count']
            bw_r = int((val_r/max_r)*bar_total_w_r) if max_r>0 else 0
            name_x_r = block_start_x_r + name_w_r - 4
            label_r = analyzer.get_player_label(p)
            dwg.add(dwg.text(label_r, insert=(name_x_r, cy_r + bar_h_r/2), fill=color, font_size=name_font_r, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
            bar_x_r = block_start_x_r + name_w_r + gap_between_r
            dwg.add(dwg.rect(insert=(bar_x_r, cy_r), size=(bar_total_w_r, bar_h_r-3), fill=PANEL_BG_COLOR, rx=2, ry=2))
            dwg.add(dwg.rect(insert=(bar_x_r, cy_r), size=(bw_r, bar_h_r-3), fill=color, rx=2, ry=2))
            label_text_r = f"{val_r:,.0f} HP ({cnt_r})"
            lx_r = bar_x_r + bar_total_w_r - 6
            dwg.add(dwg.text(label_text_r, insert=(lx_r, cy_r + bar_h_r/2), text_anchor='end', fill=TEXT_PRIMARY, font_size=value_font_r, **{'dominant-baseline':'middle'}))
            cy_r += bar_h_r + gap_r
    
    dwg.add(dwg.text('Heal Given', insert=(left_x + left_w/2, left_y + 16), text_anchor='middle', fill=TEXT_PRIMARY, font_size=12, font_weight='bold'))
    dwg.add(dwg.text('Heal Received', insert=(right_x + right_w/2, right_y + 16), text_anchor='middle', fill=TEXT_PRIMARY, font_size=12, font_weight='bold'))
    return save_svg_and_png(dwg, f'healing_summary{suffix}')


def build_damage_taken(analyzer, suffix=''):
    """Damage taken"""
    dmg = analyzer.calculate_damage_taken()
    players = sorted([p for p in dmg.keys() if dmg[p]['total_damage_taken']>0], key=lambda p: dmg[p]['total_damage_taken'], reverse=True)
    dwg = create_base(f"Damage Taken - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer)
    left_x,left_y,left_w,left_h = 12,56,726,228
    dwg.add(dwg.rect(insert=(left_x,left_y), size=(left_w,left_h), rx=6, ry=6, fill='none', stroke='none'))
    if players:
        maxd = max(dmg[p]['total_damage_taken'] for p in players)
        # Use full width: 15% for names, small gap, rest for bars
        min_margin = 24  # minimum space from panel edge
        available_w = left_w - (2 * min_margin)
        gap_between = 8
        name_w = available_w * 0.15
        bar_total_w = available_w * 0.85
        total_block_w = name_w + gap_between + bar_total_w
        block_start_x = left_x + min_margin
        
        gap=10
        num=len(players)
        bar_h = max(16, (left_h - (num+1)*gap)//num)
        cy = left_y + gap
        for p in players:
            color = analyzer.player_colors.get(p, DEFAULT_PLAYER_COLOR)
            val = dmg[p]['total_damage_taken']
            deaths = dmg[p]['deaths']
            bw = int((val/maxd)*bar_total_w) if maxd>0 else 0
            name_x = block_start_x + name_w - 4
            label = analyzer.get_player_label(p)
            dwg.add(dwg.text(label, insert=(name_x, cy + bar_h/2), fill=color, font_size=13, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
            bar_x = block_start_x + name_w + gap_between
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bar_total_w, bar_h-3), fill=PANEL_BG_COLOR, rx=2, ry=2))
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bw, bar_h-3), fill=color, rx=2, ry=2))
            label_text_dt = f"{val:,.0f}{' [DIED x'+str(deaths)+']' if deaths>0 else ''}"
            lx_dt = bar_x + bar_total_w - 6
            dwg.add(dwg.text(label_text_dt, insert=(lx_dt, cy + bar_h/2), text_anchor='end', fill=TEXT_PRIMARY, font_size=12, **{'dominant-baseline':'middle'}))
            cy += bar_h + gap
    return save_svg_and_png(dwg, f'damage_taken{suffix}')


def build_overall(analyzer, suffix=''):
    """Overall stats table"""
    dps = analyzer.calculate_dps_stats()
    healing = analyzer.calculate_healing_stats()
    dmg = analyzer.calculate_damage_taken()
    players = analyzer.players
    dwg = create_base(f"Overall Stats - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer)
    x,y,wid,hei = 12,56,726,228
    dwg.add(dwg.rect(insert=(x,y), size=(wid,hei), rx=6, ry=6, fill='none', stroke='none'))
    headers = ['Player','Total Dmg','DPS','Healing','Dmg Taken','Deaths']
    base_cols = [180,120,80,120,120,80]
    total_base = sum(base_cols)
    scaled = [max(50, int(c / total_base * wid)) for c in base_cols]
    scaled[-1] = wid - sum(scaled[:-1])

    curr_x = x
    for wcol in scaled[:-1]:
        curr_x += wcol
        dwg.add(dwg.line(start=(curr_x, y), end=(curr_x, y+hei), stroke=GRID_LINE_COLOR, stroke_width=0.8))

    header_h = 28
    data_h = max(1, hei - header_h)
    rows_count = len(players)
    row_h = int(data_h / max(1, rows_count))

    hy = y + header_h
    dwg.add(dwg.line(start=(x, hy), end=(x+wid, hy), stroke=GRID_LINE_COLOR, stroke_width=0.8))
    for i in range(1, rows_count):
        ry_line = y + header_h + i * row_h
        dwg.add(dwg.line(start=(x, ry_line), end=(x+wid, ry_line), stroke=GRID_LINE_COLOR, stroke_width=0.8))

    col_x = x
    header_y = y + header_h/2
    for i,h in enumerate(headers):
        dwg.add(dwg.text(h, insert=(col_x+8, header_y), fill=TEXT_PRIMARY, font_size=12, font_weight='bold', **{'dominant-baseline':'middle'}))
        col_x += scaled[i]

    for row_idx, p in enumerate(players):
        col_x = x
        ry = y + header_h + (row_idx + 0.5) * row_h
        player_label = analyzer.get_player_label(p)
        vals = [player_label, f"{dps[p]['total_damage']:,.0f}", f"{dps[p]['dps']:.1f}", f"{healing[p]['total_healing']:,.0f}", f"{dmg[p]['total_damage_taken']:,.0f}", f"{dmg[p]['deaths']}"]
        for i,v in enumerate(vals):
            color = analyzer.player_colors.get(p) if i==0 else TEXT_PRIMARY
            dwg.add(dwg.text(str(v), insert=(col_x+8, ry), fill=color, font_size=11, font_weight='bold' if i==0 else 'normal', **{'dominant-baseline':'middle'}))
            col_x += scaled[i]
    return save_svg_and_png(dwg, f'overall_stats{suffix}')


def build_weaponskill(analyzer, suffix=''):
    """Weaponskill details"""
    ws = analyzer.calculate_weaponskill_stats()
    dwg = create_base(f"Weaponskill Details - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer)
    x,y,wid,hei = 12,56,726,228
    dwg.add(dwg.rect(insert=(x,y), size=(wid,hei), rx=6, ry=6, fill='none', stroke='none'))
    headers = ['Player','WS','Uses','Min','Max','Avg','Total']
    base_cols = [140,180,60,60,60,60,120]
    total_base = sum(base_cols)
    scaled = [max(50, int(c / total_base * wid)) for c in base_cols]
    scaled[-1] = wid - sum(scaled[:-1])

    curr_x = x
    for wcol in scaled[:-1]:
        curr_x += wcol
        dwg.add(dwg.line(start=(curr_x, y), end=(curr_x, y+hei), stroke=GRID_LINE_COLOR, stroke_width=0.8))

    header_h = 28
    data_h = max(1, hei - header_h)
    rows_count = min(10, len(ws))
    row_h = int(data_h / max(1, rows_count))

    hy = y + header_h
    dwg.add(dwg.line(start=(x, hy), end=(x+wid, hy), stroke=GRID_LINE_COLOR, stroke_width=0.8))
    for i in range(1, rows_count):
        ry_line = y + header_h + i * row_h
        dwg.add(dwg.line(start=(x, ry_line), end=(x+wid, ry_line), stroke=GRID_LINE_COLOR, stroke_width=0.8))

    col_x = x
    header_y = y + header_h/2
    for i,h in enumerate(headers):
        dwg.add(dwg.text(h, insert=(col_x+8, header_y), fill=TEXT_PRIMARY, font_size=12, font_weight='bold', **{'dominant-baseline':'middle'}))
        col_x += scaled[i]

    for row_idx, item in enumerate(ws[:10]):
        col_x = x
        ry = y + header_h + (row_idx + 0.5) * row_h
        player_label = analyzer.get_player_label(item['Player'])
        vals = [player_label, item['Weaponskill'], item['Count'], f"{item['Min']:,.0f}", f"{item['Max']:,.0f}", f"{item['Avg']:,.0f}", f"{item['Total']:,.0f}"]
        for i,v in enumerate(vals):
            color = analyzer.player_colors.get(item['Player']) if i==0 else TEXT_PRIMARY
            dwg.add(dwg.text(str(v), insert=(col_x+8, ry), fill=color, font_size=11, **{'dominant-baseline':'middle'}))
            col_x += scaled[i]
    return save_svg_and_png(dwg, f'weaponskill_details{suffix}')


def build_key_moments(analyzer, suffix=''):
    """Key moments: deaths, victory, fight start, milestones"""
    km = analyzer.get_key_moments()

    # Convert milestones and fight start into key-moment entries so they render the same way
    milestones = analyzer.calculate_boss_hp_milestones()
    additional = []
    if milestones:
        start_t = milestones.get('start_time')
        if start_t:
            additional.append({
                'time': start_t,
                'player': '',
                'action': '',
                'type': 'Fight Start',
                'color': FIGHT_START_COLOR
            })
        for key in ('75%', '50%', '25%'):
            if key in milestones:
                additional.append({
                    'time': milestones[key],
                    'player': '',
                    'action': '',
                    'type': f'Boss HP {key}',
                    'color': TEXT_SECONDARY
                })

    combined = list(km) + additional
    combined.sort(key=lambda x: x.get('time',''), reverse=True)

    # Determine required canvas height based on number of entries and their details
    content_start_y = 56
    ry_sim = content_start_y + 12
    per_entry_heights = []
    for m in combined:
        entry_h = 16  # title line
        details = m.get('action', '')
        if details:
            detail_parts = [part.strip() for part in details.split(';') if part.strip()]
            entry_h += len(detail_parts) * 14
        entry_h += 6  # spacing after entry
        per_entry_heights.append(entry_h)

    total_entries_height = sum(per_entry_heights)
    needed_h = content_start_y + 12 + total_entries_height + 16  # bottom margin
    canvas_h = max(H, int(needed_h))

    # Create a base drawing with computed height
    dwg = create_base(f"Key Moments - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer, width=W, height=canvas_h)
    x,y,wid,hei = 12,56,W-24, canvas_h - 56 - 16
    dwg.add(dwg.rect(insert=(x,y), size=(wid,hei), rx=6, ry=6, fill='none', stroke='none'))
    ry = y + 12

    # Render all entries
    for m in combined:
        player = m.get('player', '')
        color = m.get('color') if m.get('color') else (analyzer.player_colors.get(player, TEXT_PRIMARY) if player else TEXT_SECONDARY)
        if player:
            title_txt = f"[{m.get('time')}] {m.get('type')}: {player}"
        else:
            title_txt = f"[{m.get('time')}] {m.get('type')}"

        details = m.get('action', '')
        dwg.add(dwg.text(title_txt, insert=(x+12, ry), fill=color, font_size=12, font_weight='bold', **{'dominant-baseline':'middle'}))
        ry += 16
        if details:
            detail_parts = [part.strip() for part in details.split(';') if part.strip()]
            for detail in detail_parts:
                if detail:
                    dwg.add(dwg.text(detail, insert=(x+20, ry), fill=TEXT_SECONDARY, font_size=10, **{'dominant-baseline':'middle'}))
                    ry += 14
        ry += 6

    return save_svg_and_png(dwg, f'key_moments{suffix}', width=W, height=canvas_h)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run the battle analyzer"""
    # Find all input files and group them by timestamp prefix in filename (e.g. "MM-DD-YYYY HH-MM-SS")
    all_files = [p for p in INPUT_DIR.iterdir() if p.is_file()]
    groups = defaultdict(list)
    ts_re = re.compile(r'^(\d{1,2}-\d{1,2}-\d{4} \d{2}-\d{2}-\d{2})')
    for p in all_files:
        m = ts_re.match(p.name)
        if m:
            key = m.group(1)
        else:
            # fallback to file modification timestamp
            key = datetime.fromtimestamp(p.stat().st_mtime).strftime('%m-%d-%Y %H-%M-%S')
        groups[key].append(p)

    # Build list of groups that contain at least a Battle Log and Database CSV (new format)
    valid_groups = []
    for key, files in groups.items():
        names = [f.name for f in files]
        has_battle = any('Battle Log' in n for n in names)
        has_database = any('Database' in n for n in names)
        if has_battle and has_database:
            valid_groups.append((key, files))

    if not valid_groups:
        print("ERROR: Could not find required CSV file sets in 'input/' folder!")
        print("Please ensure the 'input/' folder contains matching 'Battle Log' and 'Basic' CSVs (grouped by timestamp prefix).")
        return

    # Sort groups from older to newer by parsing the timestamp key
    def _parse_group_key(k):
        try:
            return datetime.strptime(k, '%m-%d-%Y %H-%M-%S')
        except Exception:
            # fallback: use epoch of newest file in group
            files = groups.get(k, [])
            if not files:
                return datetime.fromtimestamp(0)
            return datetime.fromtimestamp(max(f.stat().st_mtime for f in files))

    valid_groups.sort(key=lambda t: _parse_group_key(t[0]))

    # Process each group in order
    print(f"Processing {len(valid_groups)} input groups (oldest → newest)")
    generated_files = []
    for key, files in valid_groups:
        names = {f.name: f for f in files}
        # find files by pattern
        battle_file = next((f for f in files if 'Battle Log' in f.name), None)
        database_file = next((f for f in files if 'Database' in f.name), None)
        catalog_file = next((f for f in files if 'Catalog' in f.name), None)
        background_file = (INPUT_DIR / 'background.png') if (INPUT_DIR / 'background.png').exists() else None

        print(f"\nFound group: {key}")
        print(f"  Battle Log: {battle_file.name if battle_file else 'MISSING'}")
        print(f"  Database: {database_file.name if database_file else 'MISSING'}")
        if catalog_file:
            print(f"  Catalog: {catalog_file.name}")

        if not battle_file or not database_file:
            print("  → Skipping group (missing required files)")
            continue

        analyzer = FFXIBattleAnalyzer(battle_file, database_file, catalog_file, background_file)
        print(f"  Boss detected: {analyzer.boss}")
        print(f"  Players: {', '.join(analyzer.players)}")

        safe_suffix = '_' + key.replace(' ', '_')

        # Generate all images for this group
        group_generated = []
        for builder in (build_dps, build_healing, build_damage_taken, build_overall, build_weaponskill, build_key_moments):
            try:
                png_path = builder(analyzer, suffix=safe_suffix)
                if png_path:
                    group_generated.append(png_path)
                    generated_files.append(png_path)
            except TypeError:
                # builder may not accept suffix (defensive) — try without
                png_path = builder(analyzer)
                if png_path:
                    group_generated.append(png_path)
                    generated_files.append(png_path)

        # If configured, upload this group's images to Discord with a header message
        if DISCORD_WEBHOOK_URL and group_generated:
            header = f"📊 Parsing data for {analyzer.boss} {key}"
            send_discord_message(header, DISCORD_WEBHOOK_URL)
            for png_file in group_generated:
                upload_to_discord(png_file, DISCORD_WEBHOOK_URL)

    print(f'\nAll PNGs generated in {OUTPUT_DIR}')
    # Note: uploads are performed per-group above (a header message is sent, then that group's images).
    # The global upload loop was removed to avoid duplicate uploads.


if __name__ == '__main__':
    main()
