"""
FORCE — Threshold PSI Tool for Multi-Jurisdictional AML Investigations
========================================================================

Interactive Jupyter dashboard implementing a t-out-of-n threshold
Private Set Intersection (PSI) protocol for cross-border AML
deconfliction, based on Shamir's Secret Sharing (SSS) over the
Mersenne prime field 2^127 - 1.

Security model (semi-honest)
-----------------------------
All n participants, including the Leader (P0), are assumed to be
semi-honest: they follow the protocol but may attempt to infer
additional information from the messages they receive.

  - The Leader (P0) is a trusted coordinator that maintains the
    watchlist S0. P0 does not learn the contents of other
    participants' databases beyond the confirmed intersection.
  - Each participant Pi receives a single polynomial share f(i)
    per watchlist element. A single share is information-theoretically
    independent of the secret (SSS guarantee for t-1 shares).
  - Non-matching records of Pi remain local; only addresses present
    in S0 can ever produce a matching share.
  - An intersection is confirmed only when >= t participants hold the
    same address (Lagrange interpolation). Otherwise the reconstructed
    value is a uniformly random field element, revealing nothing.

Limitations
-----------
  - This is a single-process simulation: all parties run in the same
    process. A production deployment requires authenticated, encrypted
    channels between nodes and a distributed execution environment.
  - The protocol does not protect against a malicious Leader (P0) who
    deviates from the protocol. A fully malicious-secure variant would
    require verifiable secret sharing or zero-knowledge proofs.
  - HMAC tags authenticate shares against in-transit tampering but do
    not provide full malicious security.

Implementation notes
---------------------
  - Field mapping: SHA-256 XOR truncated SHA-512, masked to 127 bits.
  - Polynomial coefficients are sampled via os.urandom (CSPRNG).
  - Share generation uses a vectorised Horner evaluation (NumPy).
  - All modular arithmetic uses constant-time pow(x, -1, p).
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
import hashlib
import hmac
import os
import secrets
import time

# ---------------------------------------------------------------------------
# Cryptographic core
# ---------------------------------------------------------------------------

PRIME = 2**127 - 1
_HMAC_KEY = secrets.token_bytes(32)


def hash_to_field(element: str) -> int:
    """Map an arbitrary string to a field element in Z_p via dual-hash XOR."""
    encoded = str(element).strip().lower().encode()
    h256 = int.from_bytes(hashlib.sha256(encoded).digest(), 'big')
    h512 = int.from_bytes(hashlib.sha512(encoded).digest()[:32], 'big')
    return (h256 ^ h512) & ((1 << 127) - 1)


def batch_random_field(count: int) -> list:
    """Generate `count` CSPRNG-sampled field elements via os.urandom."""
    raw = os.urandom(16 * count)
    return [
        int.from_bytes(raw[i * 16:(i + 1) * 16], 'big') & ((1 << 127) - 1)
        for i in range(count)
    ]


def poly_eval(poly: list, x: int) -> int:
    """Evaluate a polynomial at x using Horner's method, mod PRIME."""
    result = 0
    for coef in reversed(poly):
        result = (result * x + coef) % PRIME
    return result


def make_share_tag(participant_idx: int, address_hash: int, share_value: int) -> str:
    msg = f"{participant_idx}:{address_hash}:{share_value}".encode()
    return hmac.new(_HMAC_KEY, msg, hashlib.sha256).hexdigest()


def verify_share_tag(participant_idx: int, address_hash: int,
                      share_value: int, tag: str) -> bool:
    expected = make_share_tag(participant_idx, address_hash, share_value)
    return hmac.compare_digest(expected, tag)


def lagrange_interpolation(points: list, x_target: int = 0) -> int:
    """
    Lagrange interpolation over Z_p.

    Closed-form solution for t = 2; general O(t^2) formula for t > 2.
    """
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        num0 = (-x1) % PRIME
        den0 = (x0 - x1) % PRIME
        num1 = (-x0) % PRIME
        den1 = (x1 - x0) % PRIME
        term0 = y0 * num0 % PRIME * pow(den0, -1, PRIME) % PRIME
        term1 = y1 * num1 % PRIME * pow(den1, -1, PRIME) % PRIME
        return (term0 + term1) % PRIME

    total = 0
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        num, den = 1, 1
        for j in range(n):
            if i == j:
                continue
            xj, _ = points[j]
            num = (num * (x_target - xj)) % PRIME
            den = (den * (xi - xj)) % PRIME
        inv_den = pow(den, -1, PRIME)
        term = (yi * num * inv_den) % PRIME
        total = (total + term) % PRIME
    return total


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class AMLForensicTool:
    """Interactive widget-based runner for the threshold PSI protocol."""

    def __init__(self, base_path: str = "./data"):
        self.base_path = base_path
        self.is_running = False
        self.last_results = None

        self.output = widgets.Output(
            layout={'border': '1px solid #ccc', 'padding': '10px',
                    'margin_top': '10px', 'width': 'max-content'}
        )
        self.n_input = widgets.IntText(value=5, description='Parties (n):', min=2)
        self.t_input = widgets.IntText(value=2, description='Threshold (t):', min=2)
        self.risk_filter = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05, description='Min Risk:'
        )
        self.progress = widgets.IntProgress(
            value=0, min=0, max=100, description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='98%', margin='10px 0px')
        )
        self.upload_tabs = widgets.Tab()
        self.update_tabs(None)
        self.n_input.observe(self.update_tabs, names='value')

        btn_layout = widgets.Layout(width='auto', min_width='180px', margin='5px')
        self.run_btn = widgets.Button(description="Start Investigation",
                                       button_style='success', layout=btn_layout)
        self.stop_btn = widgets.Button(description="Stop Investigation",
                                        button_style='danger', layout=btn_layout)
        self.export_btn = widgets.Button(description="Download CSV Report",
                                          button_style='info', layout=btn_layout,
                                          disabled=True)
        self.run_btn.on_click(self.execute)
        self.stop_btn.on_click(self.stop)
        self.export_btn.on_click(self.export_data)

    def update_tabs(self, change):
        n = self.n_input.value
        self.upload_tabs.children = [
            widgets.Text(
                value=os.path.join(self.base_path, f"addr_labels_J{i}.csv"),
                description='File path:',
                layout=widgets.Layout(width='95%')
            ) for i in range(n)
        ]
        for i in range(n):
            self.upload_tabs.set_title(i, f"Jurisdiction P{i}")

    def stop(self, b):
        self.is_running = False

    def export_data(self, b):
        if self.last_results is not None:
            fname = f"forensic_report_{int(time.time())}.csv"
            self.last_results.to_csv(fname, index=False)
            with self.output:
                print(f"[INFO] Report saved as: {fname}")

    def display(self):
        display(HTML(
            "<h2 style='color: #2c3e50;'>FORCE — Threshold PSI Tool</h2>"
            "<p style='color:#7f8c8d; font-size:0.85em;'>"
            "Semi-honest security model. Leader P0 is a trusted "
            "coordinator. See module docstring for the full threat "
            "model and limitations.</p>"
        ))
        display(widgets.HBox([self.n_input, self.t_input, self.risk_filter]))
        display(self.upload_tabs)
        display(widgets.HBox([self.run_btn, self.stop_btn, self.export_btn]))
        display(self.progress)
        display(self.output)

    def execute(self, b):
        self.is_running = True
        self.export_btn.disabled = True

        with self.output:
            clear_output()
            total_start = time.perf_counter()
            n = self.n_input.value
            t = self.t_input.value

            if t > n:
                print(f"[ERROR] Threshold t={t} cannot exceed n={n}.")
                self.is_running = False
                return

            self.progress.value = 0
            print(f"[INFO] Protocol parameters: n={n}, t={t}")
            print(f"[INFO] Security model: semi-honest, Leader=P0 (trusted coordinator)")
            print("-" * 60)

            # -----------------------------------------------------------
            # Phase 1 — Data loading
            # -----------------------------------------------------------
            load_start = time.perf_counter()
            dataframes = {}

            for i in range(n):
                file_path = self.upload_tabs.children[i].value.strip()
                if file_path and os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, sep=',',
                                         on_bad_lines='skip', engine='python')
                        df.columns = [c.replace(';', '').strip().lower()
                                       for c in df.columns]
                        if 'address' not in df.columns:
                            match = [c for c in df.columns if 'addr' in c]
                            if match:
                                df.rename(columns={match[0]: 'address'}, inplace=True)
                        if 'address' not in df.columns:
                            print(f"[ERROR] P{i}: no 'address' column in {file_path}")
                            continue
                        df = df.drop_duplicates(subset=['address'])
                        dataframes[i] = df
                        print(f"[OK] P{i} loaded: {len(df):,} records "
                              f"from {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"[ERROR] P{i}: could not read file — {e}")
                else:
                    print(f"[ERROR] P{i}: file not found at '{file_path}'")

            load_time = time.perf_counter() - load_start
            print(f"[TIMING] Data loading: {load_time:.4f}s")

            if len(dataframes) < t:
                print(f"[ABORT] Loaded {len(dataframes)} datasets, need >= t={t}.")
                self.is_running = False
                return

            # -----------------------------------------------------------
            # Phase 2 — Leader setup: hashing + polynomial generation
            # -----------------------------------------------------------
            self.progress.value = 20
            init_start = time.perf_counter()
            print("\n[PHASE 2] Leader P0: hashing watchlist and generating polynomials...")

            leader_idx = sorted(dataframes.keys())[0]
            leader_df = dataframes[leader_idx].copy()

            addrs = leader_df['address'].astype(str).str.strip().str.lower().tolist()
            leader_df['h'] = [hash_to_field(a) for a in addrs]
            leader_hashes = set(leader_df['h'])

            num_hashes = len(leader_hashes)
            num_coeffs = t - 1
            all_rand = batch_random_field(num_hashes * num_coeffs) if num_coeffs > 0 else []

            polys = {}
            for idx, h in enumerate(leader_hashes):
                rand_coeffs = all_rand[idx * num_coeffs:(idx + 1) * num_coeffs]
                polys[h] = [h] + rand_coeffs

            hash_list = list(leader_hashes)
            num_h = len(hash_list)

            # coeff_matrix: shape (num_hashes, t), dtype=object for 127-bit ints
            coeff_matrix = np.array(
                [[polys[h][c] for c in range(t)] for h in hash_list],
                dtype=object
            )

            # Vectorised Horner evaluation for all non-leader participants
            distributed_shares = {}
            for i in range(1, n):
                x_coord = i + 1
                result = np.zeros(num_h, dtype=object)
                for c in range(t - 1, -1, -1):
                    result = (result * x_coord + coeff_matrix[:, c]) % PRIME
                distributed_shares[i] = {}
                for idx, h in enumerate(hash_list):
                    share_val = int(result[idx])
                    tag = make_share_tag(i, h, share_val)
                    distributed_shares[i][h] = (share_val, tag)

            init_time = time.perf_counter() - init_start
            print(f"[TIMING] Leader setup: {init_time:.4f}s")

            # -----------------------------------------------------------
            # Phase 3 — Local matching and response
            # -----------------------------------------------------------
            self.progress.value = 50
            shares_start = time.perf_counter()
            print("\n[PHASE 3] Participants scanning local databases...")

            all_shares = {h: [] for h in leader_hashes}

            # Leader's own share (evaluation point x = 1)
            x_coord_leader = leader_idx + 1
            leader_result = np.zeros(num_h, dtype=object)
            for c in range(t - 1, -1, -1):
                leader_result = (leader_result * x_coord_leader + coeff_matrix[:, c]) % PRIME

            leader_df_indexed = leader_df.set_index('h')
            hash_to_address = dict(zip(leader_df['h'], leader_df['address']))

            for idx, h in enumerate(hash_list):
                share_val = int(leader_result[idx])
                row = leader_df_indexed.loc[h]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                risk = 1.0 if str(row.get('is_scam', '0')) == '1' else 0.1
                all_shares[h].append({
                    'x': x_coord_leader,
                    'y': share_val,
                    'risk': risk,
                    'label': str(row.get('description', ''))
                })

            # Non-leader participants
            for i, df in dataframes.items():
                if i == leader_idx:
                    continue
                if not self.is_running:
                    break

                df = df.copy()
                p_addrs = df['address'].astype(str).str.strip().str.lower().tolist()
                df['h'] = [hash_to_field(a) for a in p_addrs]
                matches = df[df['h'].isin(leader_hashes)]
                x_coord = i + 1

                for _, row in matches.iterrows():
                    h = row['h']
                    share_val, tag = distributed_shares[i][h]
                    if not verify_share_tag(i, h, share_val, tag):
                        print(f"[WARN] P{i}: auth FAILED for hash {h}. Skipped.")
                        continue
                    risk = 1.0 if str(row.get('is_scam', '0')) == '1' else 0.1
                    all_shares[h].append({
                        'x': x_coord,
                        'y': share_val,
                        'risk': risk,
                        'label': str(row.get('description', ''))
                    })

                print(f"[OK] P{i}: {len(matches):,} matching address(es) found and authenticated")

            shares_time = time.perf_counter() - shares_start
            print(f"[TIMING] Share computation: {shares_time:.4f}s")

            # -----------------------------------------------------------
            # Phase 4 — Threshold reconstruction
            # -----------------------------------------------------------
            self.progress.value = 80
            recon_start = time.perf_counter()
            print("\n[PHASE 4] Lagrange reconstruction and intersection verification...")

            candidates = [(h, shares) for h, shares in all_shares.items()
                           if len(shares) >= t]
            print(f"[INFO] Candidates above threshold: {len(candidates):,}")

            results = []
            for h, shares in candidates:
                if not self.is_running:
                    break

                points = [(s['x'], s['y']) for s in shares[:t]]
                reconstructed = lagrange_interpolation(points, x_target=0)

                if reconstructed != h:
                    print(f"[WARN] Reconstruction mismatch for hash {h}. Skipped.")
                    continue

                avg_risk = sum(s['risk'] for s in shares) / len(shares)
                if avg_risk < self.risk_filter.value:
                    continue

                juris = ", ".join([f"P{s['x'] - 1}" for s in shares])
                details = ", ".join(set(
                    s['label'] for s in shares
                    if s['label'] not in ('nan', '', 'None')
                ))

                orig_address = hash_to_address.get(h, str(h))

                results.append({
                    "Wallet Address": orig_address,
                    "Participating Jurisdictions": juris,
                    "Shares Used (t)": t,
                    "Total Matches": len(shares),
                    "Risk Score": round(avg_risk, 2),
                    "Details": details
                })

            recon_time = time.perf_counter() - recon_start
            print(f"[TIMING] Reconstruction & filtering: {recon_time:.4f}s")

            # -----------------------------------------------------------
            # Summary
            # -----------------------------------------------------------
            self.progress.value = 100
            total_time = time.perf_counter() - total_start

            print("\n" + "=" * 60)
            print(f"[TIMING] Total execution time: {total_time:.4f}s")
            print(f"  - Loading:        {load_time:.4f}s")
            print(f"  - Leader setup:   {init_time:.4f}s")
            print(f"  - Share compute:  {shares_time:.4f}s")
            print(f"  - Reconstruction: {recon_time:.4f}s")
            print("=" * 60)

            if results:
                self.last_results = pd.DataFrame(results)
                self.export_btn.disabled = False
                display(HTML(
                    f"<h4>Summary: found <b>{len(results)}</b> threshold "
                    f"intersection(s) (t={t}, n={n})</h4>"
                ))
                display(self.last_results)
            else:
                print("[RESULT] No intersections found above the risk threshold.")

            self.is_running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = AMLForensicTool(base_path="./data")
    app.display()
