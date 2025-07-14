import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gaussian_kde
from collections import deque
import random
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.functional import softmax
import os
from glob import glob

# ---- CONFIG ----
STATE_LEN = 15
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_MIN = 0.01
EPSILON_DEC = 5e-6
BATCH_SIZE = 32
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-4
EPOCHS = 8
WINDOW_SMOOTHING = 1
TARGET_UPDATE_FREQ = 100

# ---- ENVIRONMENT ----
class TrafficEnv:
    def __init__(self, flow, historical_mean, start_tau=0):
        self.flow = flow
        self.hist_mean = historical_mean
        self.t = STATE_LEN
        self.prev_actions = [0] * (STATE_LEN - 1)
        self.done = False
        self.start_tau = start_tau  

    def reset(self):
        self.t = STATE_LEN
        self.prev_actions = [0] * (STATE_LEN - 1)
        self.done = False
        self.build_kde_cache()
        return self.get_state()

    def build_kde_cache(self):
        self.kde_cache = {}
        J = len(self.hist_mean) // 96

        for tau in range(96):
            # Collect samples from tau-2 to tau+2 for each day
            tau_window = [(tau + offset) % 96 for offset in range(-2, 3)]  # wrap around if needed
            indices = []

            for j in range(J):  # one full day = 96 time slots
                for t in tau_window:
                    idx = j * 96 + t
                    if 0 <= idx < len(self.flow):
                        indices.append(idx)

            samples = [self.flow[i] for i in indices]

            if len(samples) > 1:
                samples = np.array(samples)
                kde = gaussian_kde(samples)
                x_grid = np.linspace(min(samples), max(samples), 500)
                y_kde = kde.evaluate(x_grid)

                peak_indices, _ = find_peaks(y_kde)
                peaks = x_grid[peak_indices]
                k_t = len(peaks)

                if k_t > 0:
                    cluster_labels = np.argmin(np.abs(samples[:, None] - peaks[None, :]), axis=1)
                    self.kde_cache[tau] = {
                        "kde": kde,
                        "samples": samples,
                        "peaks": peaks,
                        "cluster_labels": cluster_labels,
                        "k_t": k_t
                    }
                    continue

            self.kde_cache[tau] = None

    def get_state(self):
        state = []
        start_idx = self.t - STATE_LEN + 1
        for i in range(STATE_LEN):
            idx = start_idx + i
            f = self.flow[idx]
            f_bar = self.hist_mean[idx]
            if i < STATE_LEN - 1:
                a = self.prev_actions[i]
            else:
                a = 0  # No action yet for current time step
            state.append([f, f_bar, a])
        return torch.tensor(state, dtype=torch.float32)

    def get_reward(self, f_t):
        tau = (self.start_tau + self.t) % 96
        entry = self.kde_cache.get(tau)
        if entry is None:
            return 1.0

        samples = entry["samples"]
        peaks = entry["peaks"]
        cluster_labels = entry["cluster_labels"]
        k_t = entry["k_t"]
        total_points = len(samples)

        if k_t == 0:
            return 1.0

        f_t_cluster = np.argmin(np.abs(f_t - peaks))
        cluster_size = np.sum(cluster_labels == f_t_cluster)
        avg_cluster_size = total_points / k_t
        delta = cluster_size / avg_cluster_size if avg_cluster_size > 0 else 0.01
        delta = max(delta, 1e-4)

        # --- DEBUG / PLOT (OPTIONAL) ---
        # if self.t % 5000 == 0:
        #     print(f"\n[get_reward] t={self.t}, τ={tau}, f_t={f_t:.2f}, delta={delta:.6f}")
        #     print(f"  Total samples: {total_points}")
        #     print(f"  Number of clusters: {k_t}")
        #     print(f"  Cluster size for f_t: {cluster_size}")
        #     print(f"  Avg. cluster size: {avg_cluster_size:.2f}")
        # #
        #     x_grid = np.linspace(min(samples), max(samples), 500)
        #     y_kde = entry["kde"].evaluate(x_grid)
        # #
        #     import matplotlib.pyplot as plt
        #     plt.figure(figsize=(6, 3))
        #     plt.plot(x_grid, y_kde, label="KDE")
        #     plt.plot(peaks, entry["kde"].evaluate(peaks), "x", label="KDE peaks")
        #     plt.axvline(f_t, color="red", linestyle="--", label=f"f_t = {f_t:.2f}")
        #     plt.title(f"KDE (τ={tau}) — t={self.t}")
        #     plt.xlabel("Flow value")
        #     plt.ylabel("Density")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()

        return delta

    def step(self, action):
        # Prevent out-of-range before any access
        if self.t >= len(self.flow):
            self.done = True
            return None, 0.0, self.done

        f_t = self.flow[self.t]
        delta = self.get_reward(f_t)
        delta = max(delta, 1e-4)

        # Reward logic (unchanged)
        if delta < 1 and action == 1:
            reward = 1 / delta
        elif delta >= 1 and action == 0:
            reward = delta
        elif delta < 1 and action == 0:
            reward = -1 / delta
        else:
            reward = -delta

        # Update internal state
        self.prev_actions.pop(0)
        self.prev_actions.append(action)
        self.t += 1

        #  Check again after t is updated
        if self.t >= len(self.flow):
            self.done = True
            return None, reward, self.done
        # Safe to call get_state
        return self.get_state(), reward, self.done


# ---- MODEL ----
class DQNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ---- AGENT ----
class Agent:
    def __init__(self):
        self.model = DQNLSTM()
        self.target_model = DQNLSTM()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        with torch.no_grad():
            q = self.model(state.unsqueeze(0))
        return int(torch.argmax(q).item())

    def store(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        s_batch = torch.stack([b[0] for b in batch])
        a_batch = torch.tensor([b[1] for b in batch])
        r_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        s_next_batch = torch.stack([b[3] for b in batch])

        q_eval = self.model(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_model(s_next_batch).max(1)[0]
        q_target = r_batch + GAMMA * q_next

        loss = nn.MSELoss()(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DEC

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ---- SMOOTHING ----
def smooth_actions(actions, l=WINDOW_SMOOTHING):
    forward = []
    n = len(actions)
    for i in range(n):
        start = max(0, i - l)
        end = min(n, i + l + 1)
        if sum(actions[start:end]) > l:
            forward.append(1)
        else:
            forward.append(0)
    backward = []
    for i in reversed(range(n)):
        start = max(0, i - l)
        end = min(n, i + l + 1)
        if sum(actions[start:end]) > l:
            backward.append(1)
        else:
            backward.append(0)
    backward = list(reversed(backward))
    return [f & b for f, b in zip(forward, backward)]

# ---- DETECTION FUNCTION ----
def detect_anomalies_on_series(flow, hist_mean):
    env = TrafficEnv(flow, hist_mean)
    agent = Agent()

    # Train on the full series
    for epoch in range(EPOCHS):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if next_state is not None:
                agent.store(state, action, reward, next_state)
                agent.train()
                state = next_state

            total_reward += reward
            step_count += 1

            # Periodically update target network
            if step_count % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            if step_count % 1000 == 0:
                print(f"Epoch {epoch + 1} | Step {step_count} / {len(env.flow)} | Accumulated reward: {total_reward:.2f}")

        agent.update_target()

    # Run inference
    state = env.reset()
    actions = []
    done = False
    while not done:
        action = agent.act(state)
        actions.append(action)
        state, _, done = env.step(action)

    return smooth_actions(actions)


# ---- MAIN ----
if __name__ == "__main__":
    RESULTS_FILE = "RL_results.csv"
    DATA_FOLDER = "data"
    CSV_FILES = sorted(glob(os.path.join(DATA_FOLDER, "*.csv")))

    # Create CSV with headers if not exists
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            headers = ["filename"]
            for key in ["label_set_1", "label_set_2", "union", "mutual"]:
                headers += [
                    f"{key}_precision",
                    f"{key}_recall",
                    f"{key}_f1",
                    f"{key}_auc"
                ]
            f.write(",".join(headers) + "\n")

    # Loop over files
    for csv_path in CSV_FILES:
        try:
            print(f"\n=== Processing file: {csv_path} ===")
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            df["WeekdayNum"] = df["Date"].dt.weekday
            df = df[df["WeekdayNum"] < 5].reset_index(drop=True)

            df["time_of_day"] = df["Date"].dt.time
            df["TimeSlot"] = df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15
            df["tod_mean"] = df.groupby("time_of_day")["Volume"].transform("mean")

            split_idx = int(len(df) * 0.7)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            train_flow = train_df["Volume"].values.astype(np.float32)
            train_mean = train_df["tod_mean"].values.astype(np.float32)
            start_tau_train = train_df["TimeSlot"].iloc[0]

            agent = Agent()
            env = TrafficEnv(train_flow, train_mean, start_tau=start_tau_train)

            for epoch in range(EPOCHS):
                state = env.reset()
                done = False
                while not done:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    if next_state is not None:
                        agent.store(state, action, reward, next_state)
                        agent.train()
                        state = next_state
                agent.update_target()

            # Testing
            test_flow = test_df["Volume"].values.astype(np.float32)
            test_mean = test_df["tod_mean"].values.astype(np.float32)
            start_tau_test = test_df["TimeSlot"].iloc[0]
            test_env = TrafficEnv(test_flow, test_mean, start_tau=start_tau_test)

            test_env.t = STATE_LEN
            test_env.prev_actions = [0] * (STATE_LEN - 1)
            actions, scores = [], []

            while test_env.t < len(test_env.flow):
                state = test_env.get_state()
                with torch.no_grad():
                    q_values = agent.model(state.unsqueeze(0)).squeeze(0)
                    prob = softmax(q_values, dim=0)[1].item()
                    action = int(torch.argmax(q_values).item())
                actions.append(action)
                scores.append(prob)
                test_env.prev_actions.pop(0)
                test_env.prev_actions.append(action)
                test_env.t += 1

            predictions = smooth_actions(actions)

            true_1 = test_df["label_set_1"].values[STATE_LEN:]
            true_2 = test_df["label_set_2"].values[STATE_LEN:]
            union_labels = np.logical_or(true_1, true_2).astype(int)
            mutual_labels = np.logical_and(true_1, true_2).astype(int)
            prob_scores = scores[-len(true_1):]

            def get_all_metrics(y_true, y_pred, y_score):
                try:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_score)) > 1 else float('nan')
                except Exception:
                    precision, recall, f1, auc = 0.0, 0.0, 0.0, float('nan')
                return precision, recall, f1, auc

            metrics_dict = {
                "label_set_1": get_all_metrics(true_1, predictions, prob_scores),
                "label_set_2": get_all_metrics(true_2, predictions, prob_scores),
                "union": get_all_metrics(union_labels, predictions, prob_scores),
                "mutual": get_all_metrics(mutual_labels, predictions, prob_scores),
            }

            with open(RESULTS_FILE, "a") as f:
                f.write(f"{os.path.basename(csv_path)},")
                for key in ["label_set_1", "label_set_2", "union", "mutual"]:
                    p, r, f1_val, auc_val = metrics_dict[key]
                    auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "nan"
                    f.write(f"{p:.4f},{r:.4f},{f1_val:.4f},{auc_str},")
                f.write("\n")

            print(f"✓ Done. Results appended for {os.path.basename(csv_path)}")

        except Exception as e:
            print(f"✗ Failed to process {csv_path}: {e}")

