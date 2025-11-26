# taxi_dashboard.py
import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import streamlit as st

# ========= BASIC CONFIG =========
st.set_page_config(
    page_title="Taxi-v3 Q-Learning Dashboard",
    layout="wide"
)

# Small CSS tweaks: smaller metrics + tighter spacing
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3, h4 {
        margin-bottom: 0.4rem;
    }
    div[data-testid="stMetric"] {
        padding: 0.1rem 0.4rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem;
    }
    div[data-testid="stMetric"] span {
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========= PATHS & CONSTANTS =========
ENV = "Taxi-v3"
OUTPUT_DIR = "taxi_output"
Q_PATH = os.path.join(OUTPUT_DIR, "taxi_q_table.npy")
TRAIN_CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curve.png")
DEMO_CSV = os.path.join(OUTPUT_DIR, "taxi_policy_demo.csv")

LOCATIONS = ["R", "G", "Y", "B"]
ACTION_NAMES = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff",
}

# ========= LOAD Q-TABLE =========
if not os.path.exists(Q_PATH):
    st.error(f"Q-table not found at {Q_PATH}. Run taxi_with_dataset.py first.")
    st.stop()

Q = np.load(Q_PATH)

# ========= ENV IN SESSION_STATE =========
# Create the env only once and keep it in session_state
if "env" not in st.session_state:
    st.session_state.env = gym.make(ENV)

env = st.session_state.env  # use this everywhere


# ========= HELPER FUNCTIONS =========
def decode_state(state: int):
    """Decode the Taxi-v3 encoded state into helpful components."""
    taxi_row, taxi_col, passenger_idx, dest_idx = env.unwrapped.decode(int(state))
    passenger_loc = "in_taxi" if passenger_idx == 4 else LOCATIONS[passenger_idx]
    dest = LOCATIONS[dest_idx]
    return taxi_row, taxi_col, passenger_loc, dest


def build_grid(taxi_row, taxi_col, passenger_loc, dest):
    """
    Build a 5x5 grid DataFrame with:
    - R, G, Y, B landmarks
    - 🎯 destination
    - 🧍 passenger outside taxi
    - 🚕 or 🚕🧍 taxi (with passenger inside)
    """
    grid = [["" for _ in range(5)] for _ in range(5)]

    # Fixed landmark mapping
    loc_map = {
        "R": (0, 0),
        "G": (0, 4),
        "Y": (4, 0),
        "B": (4, 3),
    }

    # Landmark letters
    for name, (r, c) in loc_map.items():
        grid[r][c] = name

    # Destination mark
    dr, dc = loc_map[dest]
    grid[dr][dc] += " 🎯"

    # Passenger (waiting)
    if passenger_loc != "in_taxi":
        pr, pc = loc_map[passenger_loc]
        grid[pr][pc] += " 🧍"

    # Taxi
    taxi_icon = "🚕🧍" if passenger_loc == "in_taxi" else "🚕"
    grid[taxi_row][taxi_col] += " " + taxi_icon

    df = pd.DataFrame(grid)
    df.index = [f"Row {r}" for r in range(5)]
    df.columns = [f"Col {c}" for c in range(5)]
    return df


def greedy_action(state: int) -> int:
    return int(np.argmax(Q[int(state)]))


def reset_episode(first=False):
    """Reset env + episode stats."""
    obs, _ = env.reset()
    st.session_state.state = int(obs)
    st.session_state.step = 0
    st.session_state.total_reward = 0.0
    st.session_state.done = False
    st.session_state.last_action = "-"
    st.session_state.last_reward = 0.0
    st.session_state.steps_taken = 0
    st.session_state.penalties = 0
    st.session_state.base_fare = 0.0  # +20 when success
    st.session_state.log = []
    st.session_state.is_playing = False
    if not first:
        st.session_state.episode += 1


def step_once():
    """Single greedy step using Q-table; update all stats + log."""
    if st.session_state.done:
        st.session_state.is_playing = False
        return

    s = st.session_state.state
    a = greedy_action(s)
    next_state, reward, terminated, truncated, _ = env.step(a)
    done = bool(terminated or truncated)

    st.session_state.state = int(next_state)
    st.session_state.step += 1
    st.session_state.total_reward += float(reward)
    st.session_state.last_action = ACTION_NAMES[a]
    st.session_state.last_reward = float(reward)
    st.session_state.steps_taken += 1

    # Penalties and base fare as an interpretation of reward
    if reward == -10:
        st.session_state.penalties += 1
    if reward == 20:
        st.session_state.base_fare += 20.0

    st.session_state.log.append(
        {
            "step": st.session_state.step,
            "state": int(s),
            "action": ACTION_NAMES[a],
            "reward": float(reward),
            "next_state": int(next_state),
            "done": done,
        }
    )

    st.session_state.done = done
    if done:
        st.session_state.episode += 1
        st.session_state.is_playing = False


# ========= SESSION STATE INIT =========
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.episode = 1
    st.session_state.is_playing = False
    st.session_state.play_speed = 0.25
    reset_episode(first=True)

# ========= TITLE =========
st.title(" Taxi-v3 Q-Learning Dashboard")
st.caption(
    "Visualizing a self-driving taxi trained with **Q-Learning** in the `Taxi-v3` environment. "
    "Each **episode** is one full ride from reset until the passenger is dropped off (or the env ends)."
)

# ========= TOP LAYOUT: LEFT = GRID, RIGHT = CONTROLS & METRICS =========
left_col, right_col = st.columns([1.4, 1.1])

# ----- LEFT: GRID + LEGEND -----
with left_col:
    st.subheader("Environment View (5×5 Grid)")

    taxi_row, taxi_col, passenger_loc, dest = decode_state(st.session_state.state)
    grid_df = build_grid(taxi_row, taxi_col, passenger_loc, dest)
    st.table(grid_df)

    st.markdown("### Legend")
    st.markdown(
        """
        - 🚕 **Taxi**  
        - 🧍 **Passenger** (waiting at landmark)  
        - 🚕🧍 **Passenger inside Taxi** (after pickup)  
        - 🎯 **Destination**  
        - **R, G, Y, B** are fixed landmark locations.
        """
    )
    st.info(
        f" **Current state:** Passenger at "
        f"**{passenger_loc if passenger_loc != 'in_taxi' else 'inside taxi'}**, "
        f"Destination **{dest}**, Taxi at **row {taxi_row}, col {taxi_col}**."
    )

# ----- RIGHT: CONTROLS + EPISODE & COST (COMPACT) -----
with right_col:
    st.subheader("Controls")

    # Speed slider
    st.session_state.play_speed = st.slider(
        "Play speed (seconds per step)",
        min_value=0.05,
        max_value=1.0,
        value=st.session_state.play_speed,
        step=0.05,
    )

    # Control buttons
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("⟳ Reset"):
            reset_episode()
    with b2:
        if st.button("➡ Step"):
            step_once()
    with b3:
        if st.button("▶ Play"):
            st.session_state.is_playing = True
    with b4:
        if st.button("⏸ Pause"):
            st.session_state.is_playing = False

    # Auto-play step (will trigger on next rerun)
    if st.session_state.is_playing and not st.session_state.done:
        time.sleep(st.session_state.play_speed)
        step_once()
        st.rerun()

    # ---- Episode Info (compact) ----
    st.subheader("Episode Info")

    row1 = st.columns(3)
    with row1[0]:
        st.metric("Episode", st.session_state.episode)
    with row1[1]:
        st.metric("Step", st.session_state.step)
    with row1[2]:
        st.metric("Done", str(st.session_state.done))

    row2 = st.columns(3)
    with row2[0]:
        st.metric("Total Reward (env)", f"{st.session_state.total_reward:.1f}")
    with row2[1]:
        st.metric("Last Reward", f"{st.session_state.last_reward:.1f}")
    with row2[2]:
        st.metric("Last Action", st.session_state.last_action)

    # ---- Cost Analysis (same column, concise) ----
    st.subheader("Cost Analysis (Interpretation)")

    step_cost = st.session_state.steps_taken * 1          # 1 per step
    penalty_cost = st.session_state.penalties * 10        # 10 per penalty
    net_profit = st.session_state.base_fare - step_cost - penalty_cost

    cost_row1 = st.columns(3)
    with cost_row1[0]:
        st.metric("Steps Taken", st.session_state.steps_taken)
    with cost_row1[1]:
        st.metric("Penalties", st.session_state.penalties)
    with cost_row1[2]:
        st.metric("Base Fare (success)", f"{st.session_state.base_fare:.1f}")

    cost_row2 = st.columns(3)
    with cost_row2[0]:
        st.metric("Step Cost (1/step)", step_cost)
    with cost_row2[1]:
        st.metric("Penalty Cost (10 each)", penalty_cost)
    with cost_row2[2]:
        st.metric("Estimated Net Profit", f"{net_profit:.1f}")

# ========= BOTTOM SECTION: TABS (FULL-WIDTH) =========
st.markdown("---")
tab1, tab2, tab3 = st.tabs(
    [" Current Episode Log", " Demo Dataset", " Training Curve"]
)

with tab1:
    st.subheader("Current Episode Transitions")
    if st.session_state.log:
        log_df = pd.DataFrame(st.session_state.log)
        st.dataframe(log_df, use_container_width=True, height=300)
    else:
        st.info("Run the episode (Step or Play) to see transitions here.")

with tab2:
    st.subheader("Greedy Policy Demo Dataset")
    if os.path.exists(DEMO_CSV):
        demo_df = pd.read_csv(DEMO_CSV)
        st.write("First 100 rows from `taxi_policy_demo.csv`:")
        st.dataframe(demo_df.head(100), use_container_width=True, height=300)
        st.download_button(
            "Download full CSV",
            data=demo_df.to_csv(index=False),
            file_name="taxi_policy_demo.csv",
            mime="text/csv",
        )
    else:
        st.warning(f"{DEMO_CSV} not found. Run taxi_with_dataset.py to generate it.")

with tab3:
    st.subheader("Training Reward Curve")
    if os.path.exists(TRAIN_CURVE_PATH):
        st.image(TRAIN_CURVE_PATH, use_column_width=True)
    else:
        st.info("training_curve.png not found. Run taxi_with_dataset.py to create it.")
