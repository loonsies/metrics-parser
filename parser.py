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
        
        # Assign consistent colors to each player
        self.player_colors = self._assign_player_colors()
    
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
        """Calculate DPS statistics for all players"""
        damage_flags = ['Melee', 'WS', 'SC', 'Nuking', 'Spells']
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
        """Calculate healing statistics for all players"""
        healing_df = self.battle_log[self.battle_log['Flag'] == 'Healing']
        
        healing_stats = {}
        for player in self.players:
            player_healing = healing_df[healing_df['Player Name'] == player]
            
            # Convert healing to numeric
            healing_values = pd.to_numeric(player_healing['Damage'], errors='coerce').fillna(0)
            
            total_healing = healing_values.sum()
            heal_count = len(player_healing)
            avg_heal = total_healing / heal_count if heal_count > 0 else 0
            
            healing_stats[player] = {
                'total_healing': total_healing,
                'heal_count': heal_count,
                'avg_heal': avg_heal
            }
        
        return healing_stats
    
    def calculate_healing_received_stats(self):
        """Calculate healing received by each player (who got healed)"""
        healing_received = {p: {'total_received': 0, 'heal_count': 0} for p in self.players}

        # Prefer using the 'Basic' CSV which contains Healing Received totals per Actor/Target
        # Format: Recipient,Healer,,Healing Received,Total,Value where Actor=recipient, Target=healer
        try:
            if self.basic is not None and 'Trackable' in self.basic.columns:
                df = self.basic
                hr = df[(df['Trackable'] == 'Healing Received') & (df['Metric'] == 'Total')]
                if not hr.empty:
                    for _, row in hr.iterrows():
                        actor = row.get('Actor', '')
                        val = 0
                        try:
                            val = float(row.get('Value', 0))
                        except Exception:
                            val = 0
                        # Actor is the recipient in Basic CSV for Healing Received
                        if actor in healing_received:
                            healing_received[actor]['total_received'] += val
                
                # Also count heal instances for each recipient
                hr_attempts = df[(df['Trackable'] == 'Healing Received') & (df['Metric'] == 'Attempts')]
                if not hr_attempts.empty:
                    for _, row in hr_attempts.iterrows():
                        actor = row.get('Actor', '')
                        cnt = 0
                        try:
                            cnt = int(row.get('Value', 0))
                        except Exception:
                            cnt = 0
                        if actor in healing_received:
                            healing_received[actor]['heal_count'] += cnt
        except Exception:
            pass

        # Fallback: try to infer from battle_log Note field if basic didn't provide totals
        try:
            healing_df = self.battle_log[self.battle_log['Flag'] == 'Healing']
            if 'Note' in healing_df.columns:
                for player in self.players:
                    if healing_received[player]['total_received'] == 0:
                        received_heals = healing_df[healing_df['Note'].str.contains(player, na=False, case=False)]
                        healing_values = pd.to_numeric(received_heals['Damage'], errors='coerce').fillna(0)
                        healing_received[player]['total_received'] = healing_values.sum()
                        healing_received[player]['heal_count'] = len(received_heals)
        except Exception:
            pass

        return healing_received
    
    def calculate_damage_taken(self):
        """Calculate damage taken by each player"""
        damage_taken_stats = {}
        
        for player in self.players:
            # Get damage from basic stats
            player_basic = self.basic[
                (self.basic['Actor'] == player) & 
                (self.basic['Metric'] == 'Total') &
                (self.basic['Trackable'] == 'Total Damage Taken')
            ]
            
            if not player_basic.empty:
                total_damage_taken = player_basic['Value'].sum()
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
        """Calculate detailed weaponskill statistics for all players"""
        ws_df = self.battle_log[self.battle_log['Flag'] == 'WS']
        
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


def create_base(title, subtitle, analyzer):
    """Common base: background, title"""
    dwg = svgwrite.Drawing(size=(f'{W}px', f'{H}px'))
    try:
        dwg.attribs['font-family'] = FONT_FAMILY
    except Exception:
        pass

    _embed_font_css(dwg)

    # background image if available -> embed as data URI after processing to cover area
    if analyzer.background_image and Path(analyzer.background_image).exists():
        bg_data = _prepare_bg_data_uri(analyzer.background_image, W, H, blur_radius=IMG_BLUR_RADIUS, darken_alpha=IMG_DARKEN_ALPHA)
        if bg_data:
            dwg.add(dwg.image(href=bg_data, insert=(0, 0), size=(W, H)))
    # overlay (subtle) - color/opac configurable via .env
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill=BG_OVERLAY_COLOR, fill_opacity=BG_OVERLAY_OPACITY))
    # title
    dwg.add(dwg.text(title, insert=(W / 2, 26), text_anchor='middle', fill=TITLE_COLOR, font_size=16, font_weight='bold'))
    dwg.add(dwg.text(subtitle, insert=(W / 2, 44), text_anchor='middle', fill=SUBTITLE_COLOR, font_size=11))
    return dwg


def save_svg_and_png(dwg, name_base):
    """Save PNG only (skip SVG) and render via Playwright"""
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
            context = browser.new_context(viewport={'width':W,'height':H}, device_scale_factor=1)
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


def build_dps(analyzer):
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
            dwg.add(dwg.text(p, insert=(name_x, cy + bar_h/2), fill=color, font_size=12, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
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
        dwg.add(dwg.text(p, insert=(text_x, baseline_y), fill=color, font_size=11, **{'dominant-baseline':'middle'}))
        lx += cell_w
        col_count += 1
        if col_count >= legend_cols:
            col_count = 0
            lx = lx_start
            baseline_y += 18
    return save_svg_and_png(dwg, 'dps_summary')


def build_healing(analyzer):
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
            dwg.add(dwg.text(p, insert=(name_x, cy + bar_h/2), fill=color, font_size=name_font_size, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
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
            dwg.add(dwg.text(p, insert=(name_x_r, cy_r + bar_h_r/2), fill=color, font_size=name_font_r, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
            bar_x_r = block_start_x_r + name_w_r + gap_between_r
            dwg.add(dwg.rect(insert=(bar_x_r, cy_r), size=(bar_total_w_r, bar_h_r-3), fill=PANEL_BG_COLOR, rx=2, ry=2))
            dwg.add(dwg.rect(insert=(bar_x_r, cy_r), size=(bw_r, bar_h_r-3), fill=color, rx=2, ry=2))
            label_text_r = f"{val_r:,.0f} HP ({cnt_r})"
            lx_r = bar_x_r + bar_total_w_r - 6
            dwg.add(dwg.text(label_text_r, insert=(lx_r, cy_r + bar_h_r/2), text_anchor='end', fill=TEXT_PRIMARY, font_size=value_font_r, **{'dominant-baseline':'middle'}))
            cy_r += bar_h_r + gap_r
    
    dwg.add(dwg.text('Heal Given', insert=(left_x + left_w/2, left_y + 16), text_anchor='middle', fill=TEXT_PRIMARY, font_size=12, font_weight='bold'))
    dwg.add(dwg.text('Heal Received', insert=(right_x + right_w/2, right_y + 16), text_anchor='middle', fill=TEXT_PRIMARY, font_size=12, font_weight='bold'))
    return save_svg_and_png(dwg, 'healing_summary')


def build_damage_taken(analyzer):
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
            dwg.add(dwg.text(p, insert=(name_x, cy + bar_h/2), fill=color, font_size=13, font_weight='bold', text_anchor='end', **{'dominant-baseline':'middle'}))
            bar_x = block_start_x + name_w + gap_between
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bar_total_w, bar_h-3), fill=PANEL_BG_COLOR, rx=2, ry=2))
            dwg.add(dwg.rect(insert=(bar_x, cy), size=(bw, bar_h-3), fill=color, rx=2, ry=2))
            label_text_dt = f"{val:,.0f}{' [DIED x'+str(deaths)+']' if deaths>0 else ''}"
            lx_dt = bar_x + bar_total_w - 6
            dwg.add(dwg.text(label_text_dt, insert=(lx_dt, cy + bar_h/2), text_anchor='end', fill=TEXT_PRIMARY, font_size=12, **{'dominant-baseline':'middle'}))
            cy += bar_h + gap
    return save_svg_and_png(dwg, 'damage_taken')


def build_overall(analyzer):
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
        vals = [p, f"{dps[p]['total_damage']:,.0f}", f"{dps[p]['dps']:.1f}", f"{healing[p]['total_healing']:,.0f}", f"{dmg[p]['total_damage_taken']:,.0f}", f"{dmg[p]['deaths']}"]
        for i,v in enumerate(vals):
            color = analyzer.player_colors.get(p) if i==0 else TEXT_PRIMARY
            dwg.add(dwg.text(str(v), insert=(col_x+8, ry), fill=color, font_size=11, font_weight='bold' if i==0 else 'normal', **{'dominant-baseline':'middle'}))
            col_x += scaled[i]
    return save_svg_and_png(dwg, 'overall_stats')


def build_weaponskill(analyzer):
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
        vals = [item['Player'], item['Weaponskill'], item['Count'], f"{item['Min']:,.0f}", f"{item['Max']:,.0f}", f"{item['Avg']:,.0f}", f"{item['Total']:,.0f}"]
        for i,v in enumerate(vals):
            color = analyzer.player_colors.get(item['Player']) if i==0 else TEXT_PRIMARY
            dwg.add(dwg.text(str(v), insert=(col_x+8, ry), fill=color, font_size=11, **{'dominant-baseline':'middle'}))
            col_x += scaled[i]
    return save_svg_and_png(dwg, 'weaponskill_details')


def build_key_moments(analyzer):
    """Key moments: deaths, victory, fight start, milestones"""
    km = analyzer.get_key_moments()
    dwg = create_base(f"Key Moments - {analyzer.boss}", 'Duration: ' + analyzer.get_fight_duration()[0], analyzer)
    x,y,wid,hei = 12,56,726,228
    dwg.add(dwg.rect(insert=(x,y), size=(wid,hei), rx=6, ry=6, fill='none', stroke='none'))
    ry = y + 12

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
        for key,label_color in [('75%','#cccccc'), ('50%','#cccccc'), ('25%','#cccccc')]:
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

    for m in combined[:10]:
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
            detail_parts = [part.strip() for part in details.split(';')]
            for detail in detail_parts:
                if detail:
                    dwg.add(dwg.text(detail, insert=(x+20, ry), fill=TEXT_SECONDARY, font_size=10, **{'dominant-baseline':'middle'}))
                    ry += 14
        ry += 6
    return save_svg_and_png(dwg, 'key_moments')


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run the battle analyzer"""
    
    # Find battle log, basic, and catalog files from INPUT folder
    battle_log_files = list(INPUT_DIR.glob('*Battle Log*.csv'))
    basic_files = list(INPUT_DIR.glob('*Basic*.csv'))
    catalog_files = list(INPUT_DIR.glob('*Catalog*.csv'))
    background_files = list(INPUT_DIR.glob('background.png'))
    
    if not battle_log_files or not basic_files:
        print("ERROR: Could not find required CSV files in 'input/' folder!")
        print("Please ensure the 'input/' folder contains:")
        print("  - A file containing 'Battle Log' in the name")
        print("  - A file containing 'Basic' in the name")
        return
    
    # Select most recent files by modification time
    battle_log_file = max(battle_log_files, key=lambda p: p.stat().st_mtime)
    basic_file = max(basic_files, key=lambda p: p.stat().st_mtime)
    catalog_file = max(catalog_files, key=lambda p: p.stat().st_mtime) if catalog_files else None
    background_file = background_files[0] if background_files else None
    
    print(f"Found Battle Log: {battle_log_file.name}")
    print(f"Found Basic CSV: {basic_file.name}")
    if catalog_file:
        print(f"Found Catalog: {catalog_file.name}")
    if background_file:
        print(f"Found Background: {background_file.name}")
    
    # Initialize analyzer
    analyzer = FFXIBattleAnalyzer(
        battle_log_file,
        basic_file,
        catalog_file,
        background_file
    )
    
    print(f"\nBoss detected: {analyzer.boss}")
    print(f"Players: {', '.join(analyzer.players)}")
    print(f"Fight duration: {analyzer.get_fight_duration()[0]}")
    
    # Generate all images and collect paths
    print("\nGenerating battle summary images...")
    generated_files = []
    
    png_path = build_dps(analyzer)
    if png_path:
        generated_files.append(png_path)
    
    png_path = build_healing(analyzer)
    if png_path:
        generated_files.append(png_path)
    
    png_path = build_damage_taken(analyzer)
    if png_path:
        generated_files.append(png_path)
    
    png_path = build_overall(analyzer)
    if png_path:
        generated_files.append(png_path)
    
    png_path = build_weaponskill(analyzer)
    if png_path:
        generated_files.append(png_path)
    
    png_path = build_key_moments(analyzer)
    if png_path:
        generated_files.append(png_path)
    
    print(f'\nAll PNGs generated in {OUTPUT_DIR}')
    
    # Upload each image to Discord
    if DISCORD_WEBHOOK_URL and generated_files:
        print(f'\nUploading {len(generated_files)} images to Discord...')
        for png_file in generated_files:
            upload_to_discord(png_file, DISCORD_WEBHOOK_URL)
        print('Upload complete!')


if __name__ == '__main__':
    main()
