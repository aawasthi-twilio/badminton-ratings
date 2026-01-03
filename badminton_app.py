import streamlit as st
import sqlite3
import json
import math
import pandas as pd
from datetime import datetime

# ==========================================
# 0. DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    "initial_mu": 25.0,
    "initial_sigma": 8.333,
    "beta": 4.167,      
    "tau": 0.1,         
    "score_lever": 0.5, 
    "questions": [
        {"group": "Service", "option": "Frequent High/Out", "score": -5},
        {"group": "Service", "option": "Consistent Low", "score": 0},
        {"group": "Service", "option": "Deceptive/Flick", "score": 3},
        {"group": "Clears", "option": "Mid-court struggle", "score": -5},
        {"group": "Clears", "option": "Base-to-Base (Forehand)", "score": 2},
        {"group": "Clears", "option": "Base-to-Base (Backhand)", "score": 5},
        {"group": "Tactics", "option": "Side-by-side only", "score": -2},
        {"group": "Tactics", "option": "Rotation (Front/Back)", "score": 3},
        {"group": "Experience", "option": "Beginner", "score": -5},
        {"group": "Experience", "option": "Social/Club", "score": 0},
        {"group": "Experience", "option": "Tournament", "score": 5}
    ]
}

# ==========================================
# 1. DATABASE LAYER
# ==========================================

def get_db_connection():
    return sqlite3.connect('badminton_league.db')

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Config Table
    c.execute('''CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY,
                    settings TEXT
                )''')
    
    # Players Table
    c.execute('''CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    mu_s REAL, sigma_s REAL,
                    mu_d REAL, sigma_d REAL,
                    initial_score REAL,
                    joined_date TEXT
                )''')
    
    # Matches Table
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    match_type TEXT,
                    winners TEXT, losers TEXT,
                    score_w INTEGER, score_l INTEGER,
                    rating_changes TEXT
                )''')

    # Seed Default Config if empty
    c.execute("SELECT count(*) FROM config")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO config (id, settings) VALUES (1, ?)", (json.dumps(DEFAULT_CONFIG),))
    
    conn.commit()
    conn.close()

def get_config():
    """Load the current configuration from DB."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT settings FROM config WHERE id=1")
    res = c.fetchone()
    conn.close()
    if res:
        return json.loads(res[0])
    return DEFAULT_CONFIG

def save_config_and_reset(new_config):
    """Updates config and WIPES all data to ensure consistency."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Drop Data Tables
    c.execute("DROP TABLE IF EXISTS players")
    c.execute("DROP TABLE IF EXISTS matches")
    
    # 2. Update Config
    c.execute("UPDATE config SET settings = ? WHERE id=1", (json.dumps(new_config),))
    
    # 3. Commit and Re-init
    conn.commit()
    conn.close()
    init_db() 

# ==========================================
# 2. RATING ENGINE (Dynamic)
# ==========================================

class RatingEngine:
    def __init__(self, config):
        self.mu = config['initial_mu']
        self.sigma = config['initial_sigma']
        self.beta = config['beta']
        self.tau = config['tau']
        self.score_lever = config['score_lever']

    def get_display_rating(self, mu, sigma):
        return max(0, mu - (3 * sigma))

    def calculate_update(self, w_team, l_team, score_w, score_l):
        # Aggregate Team Stats
        mu_w = sum(p['mu'] for p in w_team)
        mu_l = sum(p['mu'] for p in l_team)
        sigma_sq_w = sum(p['sigma']**2 for p in w_team)
        sigma_sq_l = sum(p['sigma']**2 for p in l_team)

        # Dynamic Score Lever
        score_diff = score_w - score_l
        margin_multiplier = 1 + self.score_lever * math.log(score_diff + 1)

        # Prediction Math
        c = math.sqrt(sigma_sq_w + sigma_sq_l + 2 * (self.beta**2))
        diff = mu_w - mu_l
        t = diff / c
        prob = 1 / (1 + math.exp(-t)) 
        surprise = 1 - prob
        
        # Calculate Updates
        w_updates, l_updates = [], []

        for p in w_team:
            mean_delta = (p['sigma']**2 / c) * surprise * margin_multiplier
            reduction = (p['sigma']**2 / c**2)
            new_sigma = math.sqrt(p['sigma']**2 * (1 - reduction) + self.tau**2)
            w_updates.append({'mu_new': p['mu'] + mean_delta, 'sigma_new': new_sigma, 'delta': mean_delta})

        for p in l_team:
            mean_delta = (p['sigma']**2 / c) * surprise * margin_multiplier
            reduction = (p['sigma']**2 / c**2)
            new_sigma = math.sqrt(p['sigma']**2 * (1 - reduction) + self.tau**2)
            l_updates.append({'mu_new': p['mu'] - mean_delta, 'sigma_new': new_sigma, 'delta': -mean_delta})

        return w_updates, l_updates

# Load Config & Engine
init_db()
current_config = get_config()
engine = RatingEngine(current_config)

# ==========================================
# 3. DATA ACCESS HELPERS
# ==========================================

def add_player(name, initial_score_adj):
    conn = get_db_connection()
    c = conn.cursor()
    
    start_mu = current_config['initial_mu'] + initial_score_adj
    start_sigma = current_config['initial_sigma']
    
    try:
        c.execute('''INSERT INTO players 
                     (name, mu_s, sigma_s, mu_d, sigma_d, initial_score, joined_date) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (name, start_mu, start_sigma, start_mu, start_sigma, 
                   initial_score_adj, datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        st.toast(f"Player '{name}' registered!", icon="✅")
    except sqlite3.IntegrityError:
        st.error("Player name already exists.")
    finally:
        conn.close()

def get_players():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM players", conn)
    conn.close()
    return df

def save_match(match_type, winners, losers, score_w, score_l, updates_w, updates_l):
    conn = get_db_connection()
    c = conn.cursor()
    
    w_ids = [p['id'] for p in winners]
    l_ids = [p['id'] for p in losers]
    changes = {'winners': updates_w, 'losers': updates_l}
    
    col_mu = 'mu_s' if match_type == 'Singles' else 'mu_d'
    col_sigma = 'sigma_s' if match_type == 'Singles' else 'sigma_d'
    
    for i, p_id in enumerate(w_ids):
        c.execute(f"UPDATE players SET {col_mu} = ?, {col_sigma} = ? WHERE id = ?", 
                  (updates_w[i]['mu_new'], updates_w[i]['sigma_new'], p_id))
    for i, p_id in enumerate(l_ids):
        c.execute(f"UPDATE players SET {col_mu} = ?, {col_sigma} = ? WHERE id = ?", 
                  (updates_l[i]['mu_new'], updates_l[i]['sigma_new'], p_id))
        
    c.execute('''INSERT INTO matches (date, match_type, winners, losers, score_w, score_l, rating_changes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M"), match_type, 
               json.dumps(w_ids), json.dumps(l_ids), score_w, score_l, json.dumps(changes)))
    conn.commit()
    conn.close()

def get_match_history(player_id):
    conn = get_db_connection()
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
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Badminton League", layout="wide", page_icon="🏸")

st.title("🏸 Badminton League Manager")

tabs = st.tabs(["1. Register Player", "2. Log Match", "3. Profiles", "⚙️ Configuration"])

# --- TAB 1: REGISTER ---
with tabs[0]:
    st.header("New Player Registration")
    q_df = pd.DataFrame(current_config['questions'])
    col1, col2 = st.columns([1, 1])
    selections = {}
    
    with col1:
        name_input = st.text_input("Player Name")
        st.subheader("Self Evaluation")
        if q_df.empty:
            st.warning("No questions configured! Go to the Configuration tab.")
        else:
            groups = q_df['group'].unique()
            for g in groups:
                options = q_df[q_df['group'] == g]
                opt_map = dict(zip(options['option'], options['score']))
                choice = st.selectbox(f"{g}", options['option'], key=f"reg_{g}")
                selections[g] = opt_map[choice]

    with col2:
        total_adj = sum(selections.values())
        base_mu = current_config['initial_mu']
        st.metric("Base Rating (Configured)", f"{base_mu}")
        st.metric("Assessment Adjustment", f"{total_adj:+.0f}")
        st.metric("Final Estimated Skill (Mu)", f"{base_mu + total_adj}")
        
        if st.button("Register Player", type="primary"):
            if name_input:
                add_player(name_input, total_adj)
            else:
                st.warning("Enter a name.")
    
    st.divider()
    st.caption("Recent Joins")
    players = get_players()
    if not players.empty:
        st.dataframe(players[['name', 'initial_score', 'joined_date']].tail(5), hide_index=True)

# --- TAB 2: MATCHES ---
with tabs[1]:
    st.header("Log Game Result")
    players = get_players()
    
    if players.empty:
        st.info("Register players in Tab 1 to start logging matches.")
    else:
        m_type = st.radio("Match Type", ["Singles", "Doubles"], horizontal=True)
        player_map = {row['name']: row for _, row in players.iterrows()}
        p_names = list(player_map.keys())
        
        c1, c2, c3 = st.columns([2, 0.5, 2])
        
        with c1:
            st.subheader("Winners")
            w1 = st.selectbox("Winner 1", p_names, key="w1")
            w2 = None
            if m_type == "Doubles":
                w2 = st.selectbox("Winner 2", [p for p in p_names if p != w1], key="w2")
            score_w = st.number_input("Score", 0, 30, 21, key="sw")

        with c2:
            st.markdown("<br><br><h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)

        with c3:
            st.subheader("Losers")
            used = [w1]
            if w2: used.append(w2)
            avail = [p for p in p_names if p not in used]
            l1 = st.selectbox("Loser 1", avail, key="l1")
            l2 = None
            if m_type == "Doubles":
                avail2 = [p for p in avail if p != l1]
                l2 = st.selectbox("Loser 2", avail2, key="l2")
            score_l = st.number_input("Score", 0, 30, 19, key="sl")

        if st.button("Submit Match Result", use_container_width=True):
            def get_data(name, mode):
                row = player_map[name]
                if mode == 'Singles':
                    return {'id': row['id'], 'mu': row['mu_s'], 'sigma': row['sigma_s'], 'name': name}
                return {'id': row['id'], 'mu': row['mu_d'], 'sigma': row['sigma_d'], 'name': name}

            w_team = [get_data(w1, m_type)]
            if w2: w_team.append(get_data(w2, m_type))
            l_team = [get_data(l1, m_type)]
            if l2: l_team.append(get_data(l2, m_type))
            
            w_updates, l_updates = engine.calculate_update(w_team, l_team, score_w, score_l)
            save_match(m_type, w_team, l_team, score_w, score_l, w_updates, l_updates)
            
            st.success("Match Processed!")
            cols = st.columns(len(w_updates) + len(l_updates))
            all_updates = w_updates + l_updates
            all_names = [p['name'] for p in w_team] + [p['name'] for p in l_team]
            
            for i, p in enumerate(all_updates):
                new_rating = engine.get_display_rating(p['mu_new'], p['sigma_new'])
                cols[i].metric(label=all_names[i], value=f"{new_rating:.1f}", delta=f"{p['delta']:.2f}")

# --- TAB 3: PROFILES ---
with tabs[2]:
    st.header("Leaderboard & Profiles")
    players = get_players()
    if not players.empty:
        players['Singles Rating'] = players.apply(lambda x: engine.get_display_rating(x['mu_s'], x['sigma_s']), axis=1).round(1)
        players['Doubles Rating'] = players.apply(lambda x: engine.get_display_rating(x['mu_d'], x['sigma_d']), axis=1).round(1)
        
        st.dataframe(
            players[['name', 'Singles Rating', 'Doubles Rating', 'joined_date']].sort_values(by='Doubles Rating', ascending=False),
            use_container_width=True, hide_index=True
        )
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            selected = st.selectbox("View Player Details", players['name'].unique())
            p_row = players[players['name'] == selected].iloc[0]
            st.markdown(f"### {selected}")
            st.write(f"**Eval Score:** {p_row['initial_score']}")
            st.write(f"Singles: **{p_row['Singles Rating']}**")
            st.write(f"Doubles: **{p_row['Doubles Rating']}**")
        with c2:
            st.subheader("Match Log")
            hist = get_match_history(p_row['id'])
            if not hist.empty:
                st.dataframe(hist[['date', 'match_type', 'result', 'score_w', 'score_l', 'change']], hide_index=True)
            else:
                st.write("No matches played yet.")

# --- TAB 4: CONFIGURATION (ENHANCED) ---
with tabs[3]:
    st.header("⚙️ System Configuration")
    
    with st.expander("📖 Guide to Rating Physics (Read this to understand the Levers)"):
        st.markdown("""
        **1. Initial Mean (μ) & Uncertainty (σ)**
        *   **Mean (μ):** The average skill level. Standard is 25.0. 
        *   **Uncertainty (σ):** How unsure the system is. If σ is 8.0, the system thinks a player is somewhere between 1 (Novice) and 49 (Pro).
        
        **2. The Skill Gap (Beta)**
        *   **Low Beta (<3.0):** Aggressive. Assumes pure skill. Ratings swing wildly.
        *   **High Beta (>5.0):** Conservative. Assumes luck is involved. Ratings change slowly.
        
        **3. Score Weight Factor**
        *   **0.0:** A win is a win. 21-19 is the same as 21-0.
        *   **1.0:** Ruthless. A 21-0 win gives massive points compared to a close game.
        """)

    st.error("⚠️ WARNING: SAVING CONFIGURATION WILL FACTORY RESET THE DATABASE (ALL PLAYERS & MATCHES DELETED)")
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Mathematical Levers")
            new_mu = st.number_input("Initial Mean (μ)", value=current_config['initial_mu'], help="Standard: 25.0")
            new_sigma = st.number_input("Initial Uncertainty (σ)", value=current_config['initial_sigma'], help="Standard: 8.33")
            new_beta = st.number_input("Skill Gap / Luck (Beta)", value=current_config['beta'], help="Standard: 4.16")
            new_tau = st.number_input("Dynamics (Tau)", value=current_config['tau'], help="Standard: 0.1")
            new_lever = st.number_input("Score Weight Factor", value=current_config['score_lever'], help="Standard: 0.5")

        with col2:
            st.subheader("2. Self-Evaluation Questions")
            st.info("Edit the table below to change registration questions.")
            q_df = pd.DataFrame(current_config['questions'])
            edited_df = st.data_editor(
                q_df, 
                num_rows="dynamic", 
                column_config={
                    "group": st.column_config.TextColumn("Group"),
                    "option": st.column_config.TextColumn("Answer"),
                    "score": st.column_config.NumberColumn("Points")
                },
                use_container_width=True
            )

        submit = st.form_submit_button("💾 Save Configuration & RESET Database", type="primary")
        
        if submit:
            new_config = {
                "initial_mu": new_mu,
                "initial_sigma": new_sigma,
                "beta": new_beta,
                "tau": new_tau,
                "score_lever": new_lever,
                "questions": edited_df.to_dict(orient="records")
            }
            save_config_and_reset(new_config)
            st.success("System updated and database reset. Refreshing...")
            st.rerun()

    # --- ACCESSIBLE DESCRIPTION ---
    st.divider()
    st.subheader("🔎 Current System State Summary")
    
    # 1. Calculate New Player Ranges based on questions
    # We sum the MIN score of every group to get the floor, and MAX for the ceiling
    q_df_final = pd.DataFrame(current_config['questions'])
    if not q_df_final.empty:
        min_adj = q_df_final.groupby('group')['score'].min().sum()
        max_adj = q_df_final.groupby('group')['score'].max().sum()
    else:
        min_adj, max_adj = 0, 0
    
    base_mu = current_config['initial_mu']
    
    # 2. Interpret Physics
    if current_config['beta'] < 3.0: phys_desc = "Aggressive / High Volatility"
    elif current_config['beta'] > 5.0: phys_desc = "Conservative / Slow Moving"
    else: phys_desc = "Balanced / Standard Competition"
    
    if current_config['score_lever'] < 0.2: score_desc = "Winning is everything (Score margin irrelevant)"
    elif current_config['score_lever'] > 0.8: score_desc = "Blowouts matter heavily"
    else: score_desc = "Balanced (Score margin matters moderately)"

    c_a, c_b = st.columns(2)
    with c_a:
        st.info(f"""
        **League Physics:** {phys_desc}
        *   **Volatility:** The system assumes a skill gap of **{current_config['beta']}** points guarantees a win.
        *   **Scoring:** {score_desc}.
        """)
    with c_b:
        st.success(f"""
        **New Player Entry Range:**
        *   Based on the **{len(q_df_final['group'].unique()) if not q_df_final.empty else 0}** active question groups:
        *   A total beginner starts with Skill **{base_mu + min_adj:.1f}**.
        *   A pro starts with Skill **{base_mu + max_adj:.1f}**.
        *   *Note: Visible ratings start near 0 until uncertainty (σ) drops.*
        """)