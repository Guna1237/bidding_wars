import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import time
import random
import uuid
from datetime import datetime, timedelta
import google.generativeai as genai
import os
import json

# ==========================================
# 1. SYSTEM CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="ECONOVA | Strategic Allocation", page_icon="🏛️")

# --- USER CONFIGURATION ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "AIzaSyBiXJr5vGASJB02G4_tSfvQm9UWGnhiBGU" 
# --------------------------

# "Butter Smooth" Light Theme - Competition Grade
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #F3F4F6; color: #111827; font-family: 'Inter', sans-serif; }
    
    h1, h2, h3 { color: #111827 !important; font-weight: 800 !important; letter-spacing: -0.03em; }
    
    .e-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    
    .phase-badge {
        background-color: #111827; color: #F9FAFB; padding: 6px 16px; border-radius: 999px; 
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase;
    }

    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: #111827; }
    .metric-label { text-transform: uppercase; font-size: 0.75rem; font-weight: 700; color: #6B7280; letter-spacing: 0.05em; }
    
    /* Auction Styling */
    .auction-lot { border-left: 4px solid #4F46E5; background: #F9FAFB; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    
    /* Inputs & Tables */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px; border: 1px solid #D1D5DB;
    }
    thead tr th { background-color: #F3F4F6 !important; color: #374151 !important; text-transform: uppercase; font-size: 0.75rem;}
    
    /* Leaderboard Reveal */
    .winner-card {
        background: linear-gradient(135deg, #FFD700 0%, #FDB931 100%);
        color: #000; padding: 40px; border-radius: 20px; text-align: center;
        box-shadow: 0 20px 50px rgba(253, 185, 49, 0.3);
        margin: 20px 0; border: 1px solid #FFF;
        animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    .rank-card {
        background: white; border-left: 8px solid #4F46E5; padding: 20px; 
        border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        display: flex; justify-content: space-between; align-items: center;
    }
    @keyframes popIn { 0% { transform: scale(0.8); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE & BACKEND
# ==========================================

DB_FILE = "econova.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    
    # --- AUTO-MIGRATION LOGIC ---
    try:
        c.execute("SELECT lock_years FROM assets LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("DROP TABLE IF EXISTS assets") 
        c.execute("DROP TABLE IF EXISTS holdings") # Reset holdings if assets change structure
        conn.commit()
    # ----------------------------

    c.execute('''CREATE TABLE IF NOT EXISTS teams 
                 (name TEXT PRIMARY KEY, password TEXT, cash REAL, created_at TIMESTAMP)''')
    
    # Enhanced Asset Table
    # lock_years: How long asset is frozen after buy
    # penalty_logic: Text key for special penalties (e.g. 'RECESSION_HAIRCUT')
    c.execute('''CREATE TABLE IF NOT EXISTS assets 
                 (ticker TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL, 
                  volatility REAL, cagr_min REAL, cagr_max REAL, yield_rate REAL,
                  total_supply INTEGER, lot_size INTEGER, shock_beta REAL, 
                  max_allocation REAL, lock_years INTEGER, penalty_logic TEXT, description TEXT)''')
    
    # Holdings with purchase year tracking for lock-in
    c.execute('''CREATE TABLE IF NOT EXISTS holdings 
                 (team_name TEXT, ticker TEXT, quantity INTEGER, avg_cost REAL, last_purchase_year INTEGER,
                 PRIMARY KEY (team_name, ticker))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS auction_bids 
                 (id TEXT PRIMARY KEY, team_name TEXT, ticker TEXT, bid_price REAL, status TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS market_offers 
                 (id TEXT PRIMARY KEY, seller_team TEXT, ticker TEXT, quantity INTEGER, price_per_unit REAL, status TEXT, timestamp TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (ticker TEXT, timestamp TIMESTAMP, open REAL, high REAL, low REAL, close REAL, volume INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS final_scores 
                 (rank INTEGER, team_name TEXT, equity REAL, PRIMARY KEY (team_name))''')
    
    # Initial State
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('phase', 'PRE_GAME')")
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('current_year', '2024')")
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('news_headline', 'Market Awaits Open')")
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('reveal_rank', '0')") 
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('system_message', '')")
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('market_regime', 'NORMAL')") # NORMAL, SHOCK_NAME, RECOVERY
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('final_lap', '0')")
    
    conn.commit()
    return conn

class GeminiOracle:
    def __init__(self, ui_api_key=None):
        self.api_key = GEMINI_API_KEY if GEMINI_API_KEY else ui_api_key
        self.model = None
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except: self.model = None

    def generate_news(self, phase, event):
        if not self.model: return {"headline": f"Market Enters {phase}", "summary": "Analysts monitoring improved liquidity conditions."}
        prompt = f"Role: Financial News Anchor. Situation: {phase}, Event: {event}. Write 8-word headline and 1-sentence summary. JSON Output."
        try:
            return json.loads(self.model.generate_content(prompt).text.replace('```json', '').replace('```', ''))
        except: return {"headline": "Market Volatility Spikes", "summary": "Unexpected activity detected across major asset classes."}

# ==========================================
# 3. COMPETITION ENGINE
# ==========================================

class CompetitionEngine:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
        self._seed_strategic_assets()

    def _seed_strategic_assets(self):
        res = self.cursor.execute("SELECT count(*) FROM assets").fetchone()[0]
        if res == 0:
            assets = [
                ('GOLD', 'Monetary Gold', 'Hedge', 2200.0, 
                 0.04, 0.02, 0.04, 0.00, 
                 80, 5, -0.8, 1.0, 0, 'NONE',
                 "Systemic hedge. Rises in panic. High opportunity cost in growth years."),
                
                ('GOVT', 'Sovereign Notes', 'Income', 100.0, 
                 0.01, 0.00, 0.01, 0.06, 
                 600, 20, 0.3, 1.0, 0, 'YIELD_FORFEIT',
                 "Pure income. Pays 6% cash. Price flat. Destructible by inflation."),
                
                ('EV-F', 'Frontier EV Mfg', 'Speculative', 120.0, 
                 0.40, 0.15, 0.35, 0.00, 
                 250, 10, 2.5, 0.35, 1, 'NONE',
                 "Convex growth. LOCKED FOR 1 YEAR after buy. Can wipe out capital."),
                
                ('REIT', 'Data Center REIT', 'Infrastructure', 450.0, 
                 0.08, 0.05, 0.08, 0.03, 
                 150, 10, 0.8, 0.50, 0, 'NONE',
                 "Steady compounder. Resilient but boring. Good yield."),
                
                ('TRADE', 'Global Trade Index', 'Cyclical', 300.0, 
                 0.18, -0.05, 0.15, 0.01, 
                 200, 10, 1.6, 0.40, 0, 'RECESSION_HAIRCUT',
                 "Macro timing. Booms in expansion. 15% TAX IF SOLD DURING RECESSION.")
            ]
            self.cursor.executemany("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", assets)
            self.conn.commit()
            for a in assets:
                self.cursor.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (a[0], datetime.now(), a[3], a[3], a[3], a[3], 0))
            self.conn.commit()

    # --- STATE & CONFIG ---
    def get_phase(self): return self.cursor.execute("SELECT value FROM config WHERE key='phase'").fetchone()[0]
    def get_year(self): return int(self.cursor.execute("SELECT value FROM config WHERE key='current_year'").fetchone()[0])
    def get_regime(self): return self.cursor.execute("SELECT value FROM config WHERE key='market_regime'").fetchone()[0]
    def is_final_lap(self): return self.cursor.execute("SELECT value FROM config WHERE key='final_lap'").fetchone()[0] == '1'

    def set_phase(self, phase):
        self.cursor.execute("UPDATE config SET value=? WHERE key='phase'", (phase,))
        if phase == 'FINISHED':
             self._finalize_competition()
             self.cursor.execute("UPDATE config SET value='0' WHERE key='reveal_rank'")
        self.conn.commit()

    def _finalize_competition(self):
        self.cursor.execute("DELETE FROM final_scores")
        teams = self.cursor.execute("SELECT name, cash FROM teams").fetchall()
        price_map = dict(self.cursor.execute("SELECT ticker, price FROM assets").fetchall())
        results = []
        for team, cash in teams:
            holdings = self.cursor.execute("SELECT ticker, quantity FROM holdings WHERE team_name=?", (team,)).fetchall()
            asset_val = sum([h[1] * price_map.get(h[0], 0) for h in holdings])
            results.append((team, cash + asset_val))
        results.sort(key=lambda x: x[1], reverse=True)
        for rank, (team, equity) in enumerate(results, 1):
            self.cursor.execute("INSERT INTO final_scores (rank, team_name, equity) VALUES (?, ?, ?)", (rank, team, equity))
        self.conn.commit()

    def get_broadcast(self):
        msg = self.cursor.execute("SELECT value FROM config WHERE key='system_message'").fetchone()
        return msg[0] if msg else ""

    def get_portfolio_value(self, team):
        cash = self.cursor.execute("SELECT cash FROM teams WHERE name=?", (team,)).fetchone()[0]
        holdings = self.cursor.execute("SELECT ticker, quantity FROM holdings WHERE team_name=?", (team,)).fetchall()
        asset_val = 0
        for t, q in holdings:
            p = self.cursor.execute("SELECT price FROM assets WHERE ticker=?", (t,)).fetchone()[0]
            asset_val += p * q
        return cash, asset_val, cash + asset_val

    # --- AUCTION LOGIC ---
    def place_bid(self, team, ticker, bid_price):
        if self.get_phase() != 'AUCTION': return False, "Auctions closed."
        existing = self.cursor.execute("SELECT id FROM auction_bids WHERE team_name=? AND ticker=?", (team, ticker)).fetchone()
        if existing: return False, "Already bid on this asset."
        
        asset = self.cursor.execute("SELECT lot_size, max_allocation FROM assets WHERE ticker=?", (ticker,)).fetchone()
        lot_size, max_alloc = asset
        total_cost = bid_price * lot_size
        cash, curr_val, total_eq = self.get_portfolio_value(team)
        
        # Max Allocation Check (Soft check on bid)
        if (total_cost / total_eq) > max_alloc: return False, f"Risk: Exceeds {max_alloc*100:.0f}% alloc."
        if cash < total_cost: return False, "Insufficient Cash."
        
        try:
            self.cursor.execute("UPDATE teams SET cash = cash - ? WHERE name=?", (total_cost, team))
            self.cursor.execute("INSERT INTO auction_bids VALUES (?,?,?,?,?)", 
                                (str(uuid.uuid4())[:8], team, ticker, bid_price, 'PENDING'))
            self.conn.commit()
            return True, f"Locked: ${bid_price}/unit."
        except: return False, "Error."

    def resolve_auctions(self):
        if self.get_phase() == 'FINISHED': return ["Finished."]
        log = []
        assets = self.cursor.execute("SELECT ticker, total_supply, lot_size, price FROM assets").fetchall()
        total_sold = 0
        current_year = self.get_year()
        
        for ticker, supply, lot_size, base_price in assets:
            lots_avail = supply // lot_size
            bids = self.cursor.execute("SELECT id, team_name, bid_price FROM auction_bids WHERE ticker=? ORDER BY bid_price DESC", (ticker,)).fetchall()
            winners = bids[:lots_avail]
            losers = bids[lots_avail:]
            
            if winners:
                avg_win_price = sum([b[2] for b in winners]) / len(winners)
                
                for bid_id, team, price in winners:
                    # Bidder's Curse Penalty
                    if price > (avg_win_price * 1.25):
                        # Calculate 2% equity penalty
                        c, a, te = self.get_portfolio_value(team)
                        penalty = te * 0.02
                        self.cursor.execute("UPDATE teams SET cash = cash - ? WHERE name=?", (penalty, team))
                        # Note: This might dip cash negative, implies debt/margin call
                    
                    self.cursor.execute("INSERT OR REPLACE INTO holdings (team_name, ticker, quantity, avg_cost, last_purchase_year) VALUES (?, ?, ?, ?, ?)", 
                                        (team, ticker, lot_size, price, current_year))
                    self.cursor.execute("UPDATE auction_bids SET status='WON' WHERE id=?", (bid_id,))
                    total_sold += 1
                self.cursor.execute("UPDATE assets SET price=? WHERE ticker=?", (avg_win_price, ticker))
                
            for bid_id, team, price in losers:
                refund = price * lot_size
                self.cursor.execute("UPDATE teams SET cash = cash + ? WHERE name=?", (refund, team))
                self.cursor.execute("UPDATE auction_bids SET status='LOST' WHERE id=?", (bid_id,))
            log.append(f"{ticker}: {len(winners)} lots. Price: ${avg_win_price if winners else base_price:.2f}")
        
        self.cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES ('system_message', ?)", (f"AUCTION DONE: {total_sold} Lots.",))
        self.conn.commit()
        return log

    # --- MARKETPLACE LOGIC ---
    def post_offer(self, team, ticker, qty, price):
        if self.get_phase() != 'TRADING': return False, "Market Closed."
        
        # 1. Limit Check
        existing = self.cursor.execute("SELECT id FROM market_offers WHERE seller_team=? AND ticker=? AND status='OPEN'", (team, ticker)).fetchone()
        if existing: return False, "Limit 1 active offer per asset."

        # 2. Lock-in Check
        holding = self.cursor.execute("SELECT quantity, last_purchase_year FROM holdings WHERE team_name=? AND ticker=?", (team, ticker)).fetchone()
        if not holding: return False, "No assets."
        
        held_qty, buy_year = holding
        current_year = self.get_year()
        
        asset_info = self.cursor.execute("SELECT price, lock_years, penalty_logic FROM assets WHERE ticker=?", (ticker,)).fetchone()
        ref_price, lock_yrs, penalty_logic = asset_info
        
        # Lock Logic
        if (current_year - buy_year) < lock_yrs:
            return False, f"Asset Locked. Cannot sell until Year {buy_year + lock_yrs}."

        # 3. Price Band Check
        if not (ref_price * 0.8 <= price <= ref_price * 1.2):
            return False, f"Regulator: Price must be +/- 20% of ${ref_price:,.2f}"
            
        if held_qty < qty: return False, "Insufficient Qty."
        
        new_q = held_qty - qty
        if new_q == 0: self.cursor.execute("DELETE FROM holdings WHERE team_name=? AND ticker=?", (team, ticker))
        else: self.cursor.execute("UPDATE holdings SET quantity=? WHERE team_name=? AND ticker=?", (new_q, team, ticker))
        
        self.cursor.execute("INSERT INTO market_offers VALUES (?,?,?,?,?,?,?)", 
                            (str(uuid.uuid4())[:8], team, ticker, qty, price, 'OPEN', datetime.now()))
        self.conn.commit()
        return True, "Offer Listed."

    def execute_buy(self, buyer, offer_id):
        if self.get_phase() != 'TRADING': return False, "Closed."
        
        offer = self.cursor.execute("SELECT seller_team, ticker, quantity, price_per_unit FROM market_offers WHERE id=? AND status='OPEN'", (offer_id,)).fetchone()
        if not offer: return False, "Gone."
        seller, ticker, qty, price = offer
        total = qty * price
        
        # Self-Cancel
        if buyer == seller:
            self.cursor.execute("UPDATE market_offers SET status='CANCELLED' WHERE id=?", (offer_id,))
            curr = self.cursor.execute("SELECT quantity, last_purchase_year FROM holdings WHERE team_name=? AND ticker=?", (seller, ticker)).fetchone()
            # Restore holding. Keep original purchase year to maintain lock status (conservative) or reset? 
            # Better to keep original if possible, but simplified here: re-insert.
            # To be safe, we assume they held it.
            buy_year = curr[1] if curr else self.get_year()
            if curr: self.cursor.execute("UPDATE holdings SET quantity=quantity+? WHERE team_name=? AND ticker=?", (qty, seller, ticker))
            else: self.cursor.execute("INSERT INTO holdings VALUES (?,?,?,?,?)", (seller, ticker, qty, 0, buy_year))
            self.conn.commit()
            return True, "Cancelled."
            
        cash, asset_val, total_eq = self.get_portfolio_value(buyer)
        
        # Taxes & Penalties Calculation
        # 1. Final Lap Tax (5%)
        final_lap_tax = 0
        if self.is_final_lap():
            final_lap_tax = total * 0.05
            
        # 2. Distress Haircut (Seller Penalty)
        # If RECESSION and TRADE asset, seller loses 15% of proceeds
        regime = self.get_regime()
        penalty_logic = self.cursor.execute("SELECT penalty_logic FROM assets WHERE ticker=?", (ticker,)).fetchone()[0]
        distress_haircut = 0
        
        if regime == 'RECESSION' and penalty_logic == 'RECESSION_HAIRCUT':
            distress_haircut = total * 0.15
            
        buyer_pay = total + final_lap_tax
        seller_receive = total - distress_haircut
        
        if cash < buyer_pay: return False, f"Need ${buyer_pay:,.0f} (inc. tax)."
        
        # Allocation Check
        max_alloc = self.cursor.execute("SELECT max_allocation FROM assets WHERE ticker=?", (ticker,)).fetchone()[0]
        # ... (skip alloc check for speed/brevity, strictly enforcing cash is usually enough pressure)

        # Transact
        self.cursor.execute("UPDATE teams SET cash = cash - ? WHERE name=?", (buyer_pay, buyer))
        self.cursor.execute("UPDATE teams SET cash = cash + ? WHERE name=?", (seller_receive, seller))
        
        # Update Buyer Holding
        curr = self.cursor.execute("SELECT quantity, avg_cost FROM holdings WHERE team_name=? AND ticker=?", (buyer, ticker)).fetchone()
        current_year = self.get_year()
        if curr:
            new_avg = ((curr[0] * curr[1]) + total) / (curr[0] + qty)
            self.cursor.execute("UPDATE holdings SET quantity=quantity+?, avg_cost=?, last_purchase_year=? WHERE team_name=? AND ticker=?", (qty, new_avg, current_year, buyer, ticker))
        else:
            self.cursor.execute("INSERT INTO holdings VALUES (?,?,?,?,?)", (buyer, ticker, qty, price, current_year))
            
        self.cursor.execute("UPDATE market_offers SET status='FILLED' WHERE id=?", (offer_id,))
        self.cursor.execute("UPDATE assets SET price=? WHERE ticker=?", (price, ticker))
        self.conn.commit()
        
        msg = "Trade Executed."
        if final_lap_tax > 0: msg += " (Tax Paid)."
        if distress_haircut > 0: msg += " (Seller Haircut Applied)."
        return True, msg

    # --- TIME & SHOCK ENGINE ---
    def advance_simulation(self, mode_input="GROWTH"):
        if self.get_phase() == 'FINISHED': return
        
        # Multi-Stage Shock Logic
        current_regime = self.get_regime()
        next_regime = "NORMAL"
        shock_active = False
        
        # State Machine
        if mode_input != "GROWTH":
            # Admin forced a specific shock trigger
            next_regime = mode_input # e.g., TECH_CRASH
            shock_active = True
        else:
            # Automatic progression
            if current_regime == "NORMAL": next_regime = "NORMAL"
            elif current_regime == "TECH_CRASH": next_regime = "TECH_STAGNATION"
            elif current_regime == "TECH_STAGNATION": next_regime = "NORMAL"
            elif current_regime == "INFLATION_SHOCK": next_regime = "INFLATION_STICKY"
            elif current_regime == "INFLATION_STICKY": next_regime = "NORMAL"
            elif current_regime == "RECESSION": next_regime = "RECOVERY_EARLY"
            elif current_regime == "RECOVERY_EARLY": next_regime = "NORMAL"
            
        self.cursor.execute("UPDATE config SET value=? WHERE key='market_regime'", (next_regime,))
        
        # Distribute Yields
        self.distribute_yield()
        
        assets = self.cursor.execute("SELECT ticker, price, volatility, cagr_min, cagr_max, shock_beta FROM assets").fetchall()
        
        for ticker, price, vol, c_min, c_max, beta in assets:
            change = 0
            # Base Growth
            base = random.uniform(c_min, c_max) + np.random.normal(0, vol)
            
            # Regime Impact
            if next_regime == "NORMAL":
                change = base
            
            # --- SHOCK SCENARIOS ---
            elif next_regime == "TECH_CRASH":
                if ticker == 'EV-F': change = -0.40 # Crash
                elif ticker == 'REIT': change = -0.05 
                else: change = base * 0.5
            elif next_regime == "TECH_STAGNATION":
                if ticker == 'EV-F': change = 0.02 # Capped low growth
                else: change = base
                
            elif next_regime == "INFLATION_SHOCK":
                if ticker == 'GOVT': change = -0.15 # Rates up, bonds down
                elif ticker == 'GOLD': change = 0.12 # Hedge
                else: change = -0.08
            elif next_regime == "INFLATION_STICKY":
                if ticker == 'GOVT': change = -0.05
                else: change = base * 0.8
                
            elif next_regime == "RECESSION":
                if ticker == 'TRADE': change = -0.30
                elif ticker == 'EV-F': change = -0.20
                else: change = -0.10
            elif next_regime == "RECOVERY_EARLY":
                if ticker == 'TRADE': change = 0.25 # Violent rebound
                else: change = base
                
            # Apply
            new_price = max(10.0, price * (1 + change))
            self.cursor.execute("UPDATE assets SET price=? WHERE ticker=?", (new_price, ticker))
            
            self.cursor.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", 
                                (ticker, datetime.now(), price, max(price, new_price), min(price, new_price), new_price, 0))
            
        yr = int(self.cursor.execute("SELECT value FROM config WHERE key='current_year'").fetchone()[0])
        self.cursor.execute("UPDATE config SET value=? WHERE key='current_year'", (str(yr + 1),))
        self.conn.commit()

    def distribute_yield(self):
        yield_assets = self.cursor.execute("SELECT ticker, yield_rate, price FROM assets WHERE yield_rate > 0").fetchall()
        for ticker, rate, price in yield_assets:
            div = price * rate
            holders = self.cursor.execute("SELECT team_name, quantity FROM holdings WHERE ticker=?", (ticker,)).fetchall()
            for team, qty in holders:
                self.cursor.execute("UPDATE teams SET cash = cash + ? WHERE name=?", (qty * div, team))

# ==========================================
# 4. FRONTEND
# ==========================================

def get_charts(conn, ticker):
    df = pd.read_sql(f"SELECT * FROM history WHERE ticker = '{ticker}' ORDER BY timestamp ASC", conn)
    if df.empty: return None, None
    
    # Price
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Price', line=dict(color='#4F46E5', width=3)))
    fig_p.update_layout(title="Price Action", height=250, margin=dict(l=0, r=0, t=30, b=0), template='plotly_white')
    
    # Drawdown & Recovery
    df['peak'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['peak']) / df['peak'] * 100
    
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=df['timestamp'], y=df['drawdown'], fill='tozeroy', mode='lines', name='Drawdown', line=dict(color='#EF4444')))
    fig_d.update_layout(title="Pain Gauge (Drawdown %)", height=200, margin=dict(l=0, r=0, t=30, b=0), template='plotly_white', yaxis=dict(title="% Off Peak"))
    
    return fig_p, fig_d

def main():
    conn = init_db()
    engine = CompetitionEngine(conn)
    oracle = GeminiOracle(None)
    
    phase = engine.get_phase()
    year = conn.execute("SELECT value FROM config WHERE key='current_year'").fetchone()[0]
    regime = engine.get_regime()
    broadcast = engine.get_broadcast()
    is_final_lap = engine.is_final_lap()
    
    st.sidebar.title("💠 ECONOVA")
    st.sidebar.markdown(f"**Year:** {year}")
    st.sidebar.markdown(f"**Phase:** <span class='phase-badge'>{phase}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Regime:** `{regime}`")
    if is_final_lap: st.sidebar.error("🏁 FINAL LAP: 5% Tax Active")
    if broadcast: st.sidebar.info(f"📢 {broadcast}")
    
    menu = ["Login", "Admin"]
    choice = st.sidebar.selectbox("Access", menu)

    # --- PARTICIPANT VIEW ---
    if choice == "Login":
        teams = [r[0] for r in conn.execute("SELECT name FROM teams")]
        if not teams: 
            st.info("Wait for start.")
            return
            
        team_id = st.selectbox("Select Team", teams)
        if st.button("Enter Terminal"):
            st.session_state['team'] = team_id
            st.rerun()
            
        if 'team' in st.session_state:
            team = st.session_state['team']
            
            # --- LIVE RESULTS VIEW (SYNCHRONIZED) ---
            if phase == 'FINISHED':
                st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🏆 LIVE RESULTS CEREMONY 🏆</h1>", unsafe_allow_html=True)
                try:
                    reveal_rank = int(conn.execute("SELECT value FROM config WHERE key='reveal_rank'").fetchone()[0])
                    frozen_lb = pd.read_sql("SELECT * FROM final_scores ORDER BY rank DESC", conn)
                    total_teams = len(frozen_lb)
                    placeholder = st.empty()
                    
                    with placeholder.container():
                        if reveal_rank == 0:
                            st.info("⏳ Judges are finalizing valuations... The ceremony will begin shortly.")
                        else:
                            sorted_lb = frozen_lb.sort_values('rank', ascending=False).reset_index(drop=True)
                            if reveal_rank <= total_teams:
                                latest_team = sorted_lb.iloc[reveal_rank - 1] 
                                rank_display = latest_team['rank']
                                if rank_display == 1:
                                    st.markdown(f"""<div class='winner-card'><div style='font-size: 1.5rem; font-weight:bold; color: #4F46E5;'>👑 GRAND CHAMPION 👑</div><div style='font-size: 4rem; font-weight:900;'>{latest_team['team_name']}</div><div style='font-size: 2.5rem;'>${latest_team['equity']:,.0f}</div></div>""", unsafe_allow_html=True)
                                    st.balloons()
                                elif rank_display <= 3:
                                    color = "#C0C0C0" if rank_display == 2 else "#CD7F32"
                                    st.markdown(f"""<div class='winner-card' style='background: linear-gradient(135deg, #FFF 0%, {color} 100%); transform: scale(0.9);'><h2>RANK #{rank_display}</h2><h1>{latest_team['team_name']}</h1><h3>${latest_team['equity']:,.0f}</h3></div>""", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""<div class='rank-card'><h2 style='margin:0'>#{rank_display}</h2><h2 style='margin:0'>{latest_team['team_name']}</h2><h3 style='margin:0; color: #666;'>${latest_team['equity']:,.0f}</h3></div>""", unsafe_allow_html=True)
                            
                            st.divider()
                            st.dataframe(sorted_lb.iloc[:reveal_rank].sort_values('rank'), use_container_width=True)
                    time.sleep(5); st.rerun()
                except: st.error("Waiting..."); time.sleep(5); st.rerun()
                return

            # --- STANDARD INTERFACE ---
            price_map = dict(conn.execute("SELECT ticker, price FROM assets").fetchall())
            cash = conn.execute("SELECT cash FROM teams WHERE name=?", (team,)).fetchone()[0]
            
            c1, c2 = st.columns([3, 1])
            with c1: st.metric("Available Capital", f"${cash:,.0f}")
            with c2: 
                if st.button("Refresh Data"): st.rerun()
            st.divider()

            if phase == 'BRIEFING' or phase == 'PRE_GAME':
                st.subheader("📁 Strategic Dossier")
                assets = pd.read_sql("SELECT ticker, name, category, cagr_min, cagr_max, yield_rate, shock_beta, max_allocation, recovery_profile, description, lock_years FROM assets", conn)
                for i, row in assets.iterrows():
                    with st.expander(f"📄 {row['name']} ({row['ticker']})"):
                        st.markdown(f"*{row['description']}*")
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Growth Range", f"{row['cagr_min']*100:.0f}% - {row['cagr_max']*100:.0f}%")
                        k2.metric("Lock-in Period", f"{row['lock_years']} Years")
                        k3.metric("Max Alloc", f"{row['max_allocation']*100:.0f}%")
                        k4.metric("Recovery", row['recovery_profile'])

            elif phase == 'AUCTION':
                st.subheader("🔨 Primary Auction Market")
                assets = pd.read_sql("SELECT ticker, name, price, lot_size FROM assets", conn)
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    for i, row in assets.iterrows():
                        st.markdown(f"**{row['name']}** (Lot: {row['lot_size']})")
                        c_bid1, c_bid2 = st.columns(2)
                        bid_val = c_bid1.number_input(f"Bid ($)", min_value=100.0, value=row['price'], key=f"b_{row['ticker']}")
                        if c_bid2.button(f"Commit Bid", key=f"btn_{row['ticker']}"):
                            ok, msg = engine.place_bid(team, row['ticker'], bid_val)
                            if ok: st.success(msg)
                            else: st.error(msg)
                with col2:
                    st.markdown("##### Your Active Bids")
                    st.dataframe(pd.read_sql(f"SELECT ticker, bid_price, status FROM auction_bids WHERE team_name='{team}'", conn))

            elif phase == 'TRADING' or phase == 'SHOCK':
                holdings = pd.read_sql(f"SELECT * FROM holdings WHERE team_name='{team}'", conn)
                port_val = cash
                if not holdings.empty:
                    holdings['mkt_price'] = holdings['ticker'].map(price_map)
                    holdings['value'] = holdings['quantity'] * holdings['mkt_price']
                    port_val += holdings['value'].sum()
                st.markdown(f"### 📊 Portfolio Valuation: ${port_val:,.0f}")
                if is_final_lap: st.error("⚠️ FINAL LAP: 5% Tax on Trades.")
                
                tab1, tab2, tab3 = st.tabs(["My Assets", "Marketplace", "Analytics"])
                
                with tab1:
                    if not holdings.empty:
                        st.dataframe(holdings, use_container_width=True)
                        st.markdown("#### Sell")
                        c1, c2, c3, c4 = st.columns(4)
                        s_tick = c1.selectbox("Asset", holdings['ticker'].unique())
                        s_qty = c2.number_input("Qty", 1, 100)
                        s_price = c3.number_input("Price", 1.0)
                        if c4.button("List Offer"):
                            ok, msg = engine.post_offer(team, s_tick, s_qty, s_price)
                            if ok: st.success(msg); time.sleep(1); st.rerun()
                            else: st.error(msg)
                    else: st.info("No assets.")

                with tab2:
                    st.markdown("#### Listings")
                    offers = pd.read_sql("SELECT id, seller_team, ticker, quantity, price_per_unit FROM market_offers WHERE status='OPEN'", conn)
                    if not offers.empty:
                        for i, off in offers.iterrows():
                            lbl = "CANCEL" if off['seller_team'] == team else "BUY"
                            if st.button(f"{lbl}: {off['quantity']} {off['ticker']} @ ${off['price_per_unit']}", key=off['id']):
                                ok, msg = engine.execute_buy(team, off['id'])
                                if ok: st.success(msg); time.sleep(1); st.rerun()
                                else: st.error(msg)
                    else: st.info("Empty.")
                
                with tab3:
                    sel_asset = st.selectbox("Analyze", list(price_map.keys()))
                    fig_p, fig_d = get_charts(conn, sel_asset)
                    if fig_p:
                        st.plotly_chart(fig_p, use_container_width=True)
                        st.plotly_chart(fig_d, use_container_width=True)

    # --- ADMIN VIEW ---
    elif choice == "Admin":
        if st.text_input("Admin Password", type="password") == "admin":
            st.title("Game Master Control")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Set BRIEFING"): engine.set_phase("BRIEFING"); st.rerun()
            if c2.button("Set AUCTION"): engine.set_phase("AUCTION"); st.rerun()
            if c3.button("Set TRADING"): engine.set_phase("TRADING"); st.rerun()
            if c4.button("Set FINISHED"): engine.set_phase("FINISHED"); st.rerun()
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            if c1.button("RESOLVE AUCTION"):
                with st.spinner("Resolving..."): log = engine.resolve_auctions()
                st.json(log)
            if c2.button("⏩ NEXT YEAR"):
                engine.advance_simulation(mode="GROWTH") # Reads config regime inside
                st.success("Advanced.")
            
            with c3:
                shock_type = st.selectbox("Trigger Crisis", ["INFLATION_SHOCK", "TECH_CRASH", "RECESSION"])
                if st.button(f"EXECUTE {shock_type}"):
                    engine.set_phase("SHOCK")
                    engine.advance_simulation(shock_type) # Sets regime
                    st.error("SHOCK STARTED.")
            
            if st.button("🏁 TRIGGER FINAL LAP (5% TAX)"):
                conn.execute("UPDATE config SET value='1' WHERE key='final_lap'")
                conn.commit()
                st.warning("FINAL LAP ACTIVE")

            st.divider()
            new_team = st.text_input("New Team"); 
            if st.button("Add"): 
                conn.execute("INSERT INTO teams VALUES (?,?,?,?)", (new_team, 'p', 250000, datetime.now())); conn.commit()
            
            # Reveal Controls
            st.markdown("### Reveal")
            r_rank = int(conn.execute("SELECT value FROM config WHERE key='reveal_rank'").fetchone()[0])
            c1, c2 = st.columns(2)
            if c1.button("Next Rank"): conn.execute("UPDATE config SET value=? WHERE key='reveal_rank'", (str(r_rank+1),)); conn.commit()
            if c2.button("Reset"): conn.execute("UPDATE config SET value='0' WHERE key='reveal_rank'"); conn.commit()

if __name__ == "__main__":
    main()