import streamlit as st
import sqlite3
import json
import math
import pandas as pd
from datetime import datetime

# ==========================================
# 1. BAYESIAN RATING ENGINE
# ==========================================

class RatingEngine:
    """
    Manages the mathematics of the rating system.
    """
    def __init__(self, initial_mu=25.0, initial_sigma=8.333, beta=4.167, tau=0.1):
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.beta = beta  # The skill gap required for ~80% win probability
        self.tau = tau    # Dynamics: Prevents uncertainty from hitting zero

    def get_conservative_rating(self, mu, sigma):
        """
        Returns the user-facing rating (Mu - 3*Sigma).
        New players start near 0. Proven players rise.
        """
        return max(0, mu - (3 * sigma))

    def calculate_update(self, w_team, l_team, score_w, score_l):
        """
        Calculates new ratings for a match (1v1 or 2v2).
        w_team/l_team: List of dicts {'mu', 'sigma'}
        """
        # 1. Aggregate Team Parameters (Sum of Means, Sum of Variances)
        mu_w = sum(p['mu'] for p in w_team)
        mu_l = sum(p['mu'] for p in l_team)
        sigma_sq_w = sum(p['sigma']**2 for p in w_team)
        sigma_sq_l = sum(p['sigma']**2 for p in l_team)

        # 2. Score Lever: Logarithmic Multiplier
        # 21-19 (Diff 2) -> ~1.2x
        # 21-5  (Diff 16)-> ~2.2x
        score_diff = score_w - score_l
        margin_multiplier = 1 + 0.45 * math.log(score_diff + 1)

        # 3. Prediction & Surprise
        # c = Total uncertainty in the match
        c = math.sqrt(sigma_sq_w + sigma_sq_l + 2 * (self.beta**2))
        diff = mu_w - mu_l
        
        # t = Normalized skill difference
        t = diff / c
        # Expected win probability
        prob = 1 / (1 + math.exp(-t)) 
        
        # Surprise: High surprise if underdog wins
        surprise = 1 - prob
        
        # 4. Calculate Updates
        w_updates = []
        l_updates = []

        # Update Winners
        for p in w_team:
            # Credit Assignment: Players with higher uncertainty get larger updates
            mean_delta = (p['sigma']**2 / c) * surprise * margin_multiplier
            
            # Uncertainty shrinks (learning occurred)
            reduction = (p['sigma']**2 / c**2)
            new_sigma = math.sqrt(p['sigma']**2 * (1 - reduction) + self.tau**2)
            
            w_updates.append({
                'mu_new': p['mu'] + mean_delta, 
                'sigma_new': new_sigma,
                'delta': mean_delta
            })

        # Update Losers
        for p in l_team:
            mean_delta = (p['sigma']**2 / c) * surprise * margin_multiplier
            reduction = (p['sigma']**2 / c**2)
            new_sigma = math.sqrt(p['sigma']**2 * (1 - reduction) + self.tau**2)
            
            l_updates.append({
                'mu_new': p['mu'] - mean_delta, 
                'sigma_new': new_sigma,
                'delta': -mean_delta
            })

        return w_updates, l_updates

engine = RatingEngine()

# ==========================================
# 2. DATABASE (SQLite)
# ==========================================

def init_db():
    """Initialize database with separate Singles/Doubles profiles."""
    conn = sqlite3.connect('badminton_ratings.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    mu_s REAL, sigma_s REAL,
                    mu_d REAL, sigma_d REAL,
                    initial_answers TEXT,
                    joined_date TEXT
                )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    match_type TEXT,
                    winners TEXT,
                    losers TEXT,
                    score_w INTEGER,
                    score_l INTEGER,
                    rating_changes TEXT
                )''')
    conn.commit()
    conn.close()

def add_player(name, answers, initial_mu_adj):
    conn = sqlite3.connect('badminton_ratings.db')
    c = conn.cursor()
    
    base_mu = 25.0
    start_mu = base_mu + initial_mu_adj
    start_sigma = 8.333 # High uncertainty for everyone initially
    
    try:
        # Initialize both Singles (_s) and Doubles (_d) with the same prior
        c.execute('''INSERT INTO players 
                     (name, mu_s, sigma_s, mu_d, sigma_d, initial_answers, joined_date) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (name, start_mu, start_sigma, start_mu, start_sigma, 
                   json.dumps(answers), datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        st.success(f"Player '{name}' registered with Skill Estimate: {start_mu}")
    except sqlite3.IntegrityError:
        st.error("Player name already exists.")
    finally:
        conn.close()

def get_players():
    conn = sqlite3.connect('badminton_ratings.db')
    df = pd.read_sql_query("SELECT * FROM players", conn)
    conn.close()
    return df

def save_match(match_type, winners, losers, score_w, score_l, updates_w, updates_l):
    conn = sqlite3.connect('badminton_ratings.db')
    c = conn.cursor()
    
    w_ids = [p['id'] for p in winners]
    l_ids = [p['id'] for p in losers]
    changes = {'winners': updates_w, 'losers': updates_l}
    
    # Dynamically select columns based on match type
    col_mu = 'mu_s' if match_type == 'Singles' else 'mu_d'
    col_sigma = 'sigma_s' if match_type == 'Singles' else 'sigma_d'
    
    # Update Winners
    for i, p_id in enumerate(w_ids):
        c.execute(f"UPDATE players SET {col_mu} = ?, {col_sigma} = ? WHERE id = ?", 
                  (updates_w[i]['mu_new'], updates_w[i]['sigma_new'], p_id))
    # Update Losers
    for i, p_id in enumerate(l_ids):
        c.execute(f"UPDATE players SET {col_mu} = ?, {col_sigma} = ? WHERE id = ?", 
                  (updates_l[i]['mu_new'], updates_l[i]['sigma_new'], p_id))
        
    # Log Match
    c.execute('''INSERT INTO matches (date, match_type, winners, losers, score_w, score_l, rating_changes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M"), match_type, 
               json.dumps(w_ids), json.dumps(l_ids), score_w, score_l, json.dumps(changes)))
    
    conn.commit()
    conn.close()

def get_match_history(player_id):
    conn = sqlite3.connect('badminton_ratings.db')
    df = pd.read_sql_query("SELECT * FROM matches ORDER BY id DESC", conn)
    conn.close()
    
    history = []
    for _, row in df.iterrows():
        winners = json.loads(row['winners'])
        losers = json.loads(row['losers'])
        
        if player_id in winners or player_id in losers:
            row_data = row.to_dict()
            changes = json.loads(row['rating_changes'])
            
            if player_id in winners:
                idx = winners.index(player_id)
                delta = changes['winners'][idx]['delta']
                res = "Win"
            else:
                idx = losers.index(player_id)
                delta = changes['losers'][idx]['delta']
                res = "Loss"
                
            row_data['result'] = res
            row_data['change'] = round(delta, 2)
            history.append(row_data)
            
    return pd.DataFrame(history)

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Badminton Ratings", layout="wide")
init_db()

st.title("🏸 Bayesian Badminton Rater")

tabs = st.tabs(["1. Register Player", "2. Log Match", "3. Profiles & Leaderboard"])

# --- TAB 1: REGISTER ---
with tabs[0]:
    st.header("New Player Registration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        name_input = st.text_input("Player Name")
        st.subheader("Self Evaluation")
        st.info("These questions set your initial skill level (Prior).")
        
        # Questions mapped to skill adjustments
        q1 = st.selectbox("Serve", ["High/Out frequently (-5)", "Consistent Low (0)", "Deceptive/Flick (+3)"], index=1)
        q2 = st.selectbox("Clears", ["Mid-court only (-5)", "Base-to-Base Forehand (+2)", "Base-to-Base Backhand (+5)"], index=1)
        q3 = st.selectbox("Tactics", ["Side-by-side only (-2)", "Rotation (Front/Back) (+3)"], index=0)
        q4 = st.selectbox("Experience", ["Beginner (-5)", "Social/Club (0)", "Tournament (+5)"], index=1)

    with col2:
        # Calculate Preview
        score = 0
        if "High" in q1: score -= 5
        elif "Deceptive" in q1: score += 3
        
        if "Mid" in q2: score -= 5
        elif "Backhand" in q2: score += 5
        else: score += 2
        
        if "Side" in q3: score -= 2
        else: score += 3
        
        if "Beginner" in q4: score -= 5
        elif "Tournament" in q4: score += 5
        
        st.metric("Estimated Initial Skill (Mu)", f"{25 + score}")
        
        if st.button("Register Player"):
            if name_input:
                answers = {"Serve": q1, "Clears": q2, "Tactics": q3, "Exp": q4}
                add_player(name_input, answers, score)
            else:
                st.warning("Please enter a name.")
    
    st.divider()
    st.write("Recent Players:")
    players = get_players()
    if not players.empty:
        st.dataframe(players[['name', 'joined_date']].tail(5))

# --- TAB 2: LOG MATCH ---
with tabs[1]:
    st.header("Log Game Result")
    players = get_players()
    
    if players.empty:
        st.warning("No players found.")
    else:
        m_type = st.radio("Match Type", ["Singles", "Doubles"])
        player_map = {row['name']: row for _, row in players.iterrows()}
        p_names = list(player_map.keys())
        
        col_w, col_l = st.columns(2)
        
        with col_w:
            st.subheader("Winners")
            w1 = st.selectbox("Winner 1", p_names, key="w1")
            w2 = None
            if m_type == "Doubles":
                w2 = st.selectbox("Winner 2", [p for p in p_names if p != w1], key="w2")
            score_w = st.number_input("Winner Score", 0, 30, 21)

        with col_l:
            st.subheader("Losers")
            used = [w1]
            if w2: used.append(w2)
            avail = [p for p in p_names if p not in used]
            l1 = st.selectbox("Loser 1", avail, key="l1")
            l2 = None
            if m_type == "Doubles":
                avail2 = [p for p in avail if p != l1]
                l2 = st.selectbox("Loser 2", avail2, key="l2")
            score_l = st.number_input("Loser Score", 0, 30, 15)

        if st.button("Calculate & Submit"):
            # Prepare data structure for the engine
            def get_data(name, mode):
                row = player_map[name]
                if mode == 'Singles':
                    return {'id': row['id'], 'mu': row['mu_s'], 'sigma': row['sigma_s'], 'name': name}
                return {'id': row['id'], 'mu': row['mu_d'], 'sigma': row['sigma_d'], 'name': name}

            w_team = [get_data(w1, m_type)]
            if w2: w_team.append(get_data(w2, m_type))
            l_team = [get_data(l1, m_type)]
            if l2: l_team.append(get_data(l2, m_type))
            
            # RUN ENGINE
            w_updates, l_updates = engine.calculate_update(w_team, l_team, score_w, score_l)
            
            # Save
            save_match(m_type, w_team, l_team, score_w, score_l, w_updates, l_updates)
            
            st.success("Match Recorded!")
            cols = st.columns(len(w_updates) + len(l_updates))
            
            for i, p in enumerate(w_updates):
                name = w_team[i]['name']
                new_r = engine.get_conservative_rating(p['mu_new'], p['sigma_new'])
                cols[i].metric(f"{name} (Win)", f"{new_r:.1f}", f"+{p['delta']:.2f} Skill")
                
            for i, p in enumerate(l_updates):
                name = l_team[i]['name']
                new_r = engine.get_conservative_rating(p['mu_new'], p['sigma_new'])
                cols[len(w_updates)+i].metric(f"{name} (Loss)", f"{new_r:.1f}", f"{p['delta']:.2f} Skill")

# --- TAB 3: PROFILES ---
with tabs[2]:
    st.header("Player Profiles")
    players = get_players()
    
    if not players.empty:
        # Calculate display ratings
        players['Singles Rating'] = players.apply(lambda x: engine.get_conservative_rating(x['mu_s'], x['sigma_s']), axis=1).round(1)
        players['Doubles Rating'] = players.apply(lambda x: engine.get_conservative_rating(x['mu_d'], x['sigma_d']), axis=1).round(1)
        
        st.dataframe(players[['name', 'Singles Rating', 'Doubles Rating', 'joined_date']], use_container_width=True)
        
        st.divider()
        st.subheader("Player Details")
        selected = st.selectbox("Select Player", players['name'].unique())
        
        p_row = players[players['name'] == selected].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Singles Skill (Mu)", f"{p_row['mu_s']:.1f}")
        c2.metric("Singles Uncertainty", f"± {p_row['sigma_s']*3:.1f}")
        c3.metric("Doubles Skill (Mu)", f"{p_row['mu_d']:.1f}")
        c4.metric("Doubles Uncertainty", f"± {p_row['sigma_d']*3:.1f}")
        
        try:
            st.caption("Initial Self-Evaluation Answers:")
            st.json(json.loads(p_row['initial_answers']))
        except: pass
        
        st.subheader("Match History")
        hist = get_match_history(p_row['id'])
        if not hist.empty:
            st.dataframe(hist[['date', 'match_type', 'result', 'score_w', 'score_l', 'change']], use_container_width=True)
        else:
            st.info("No matches played yet.")
