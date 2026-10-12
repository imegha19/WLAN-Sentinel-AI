
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "synthetic_awid3_style_demo.csv"
SEED = 20261012
ROWS = 12_000


def generate_dataset(rows=ROWS, seed=SEED):
    rng = np.random.default_rng(seed)
    classes = [
        ("Normal", 0.62),
        ("Deauth", 0.09),
        ("Evil_Twin", 0.07),
        ("KRACK_like", 0.045),
        ("KR00K_like", 0.045),
        ("SSDP", 0.045),
        ("SQL_Injection", 0.04),
        ("Malware_like", 0.045),
    ]
    labels = rng.choice([item[0] for item in classes],
                        size=rows, p=[item[1] for item in classes])
    records = []

    for label in labels:
        frame_len = int(np.clip(rng.normal(750, 320), 64, 2304))
        duration = float(np.clip(rng.gamma(2.2, 180), 1, 3000))
        retry = int(rng.random() < 0.07)
        protected = int(rng.random() < 0.78)
        signal = int(np.clip(rng.normal(-55, 13), -95, -20))
        channel = int(rng.choice([1, 6, 11, 36, 40, 44, 149, 157]))
        packet_rate = float(np.clip(rng.gamma(2.5, 8), 0.1, 250))
        bytes_rate = float(np.clip(rng.lognormal(8.0, 1.0), 10, 250000))
        tcp_syn = int(rng.poisson(1.2))
        dns_queries = int(rng.poisson(2.0))
        http_requests = int(rng.poisson(2.8))
        eapol_frames = int(rng.poisson(0.6))
        deauth_frames = int(rng.poisson(0.2))
        disassoc_frames = int(rng.poisson(0.2))
        unique_dest = int(np.clip(rng.poisson(3.0), 1, 80))
        failed_auth = int(rng.poisson(0.2))
        wlan_type = int(rng.choice([0, 1, 2], p=[0.12, 0.68, 0.20]))
        wlan_subtype = int(rng.choice([0, 4, 8, 10, 12, 24, 28]))

        if label == "Deauth":
            deauth_frames += int(rng.integers(8, 65))
            disassoc_frames += int(rng.integers(2, 25))
            packet_rate = float(np.clip(packet_rate + rng.uniform(25, 130), 0.1, 250))
            retry = int(rng.random() < 0.5)
            wlan_type, wlan_subtype = 0, 12
        elif label == "Evil_Twin":
            failed_auth += int(rng.integers(2, 18))
            eapol_frames += int(rng.integers(2, 20))
            unique_dest += int(rng.integers(1, 12))
            protected = int(rng.random() < 0.3)
        elif label == "KRACK_like":
            eapol_frames += int(rng.integers(3, 16))
            retry = int(rng.random() < 0.65)
            protected = int(rng.random() < 0.5)
        elif label == "KR00K_like":
            protected = int(rng.random() < 0.2)
            frame_len = int(np.clip(rng.normal(250, 140), 64, 1200))
            retry = int(rng.random() < 0.3)
        elif label == "SSDP":
            packet_rate = float(np.clip(packet_rate + rng.integers(5, 40), 0.1, 250))
            unique_dest += int(rng.integers(2, 15))
            bytes_rate = float(np.clip(bytes_rate * rng.uniform(1.5, 6), 10, 250000))
        elif label == "SQL_Injection":
            http_requests += int(rng.integers(3, 30))
            tcp_syn += int(rng.integers(1, 8))
            bytes_rate = float(np.clip(bytes_rate * rng.uniform(1.2, 4), 10, 250000))
        elif label == "Malware_like":
            unique_dest += int(rng.integers(4, 25))
            dns_queries += int(rng.integers(4, 25))
            bytes_rate = float(np.clip(bytes_rate * rng.uniform(1.5, 5), 10, 250000))
            tcp_syn += int(rng.integers(2, 12))

        records.append({
            "frame_length_bytes": frame_len,
            "wlan_frame_type": wlan_type,
            "wlan_subtype": wlan_subtype,
            "duration_ms": round(duration, 3),
            "retry_flag": retry,
            "protected_frame": protected,
            "signal_dbm": signal,
            "channel": channel,
            "packet_rate": round(packet_rate, 3),
            "bytes_per_second": round(bytes_rate, 3),
            "tcp_syn_count": tcp_syn,
            "dns_query_count": dns_queries,
            "http_request_count": http_requests,
            "eapol_frame_count": eapol_frames,
            "deauth_frame_count": deauth_frames,
            "disassoc_frame_count": disassoc_frames,
            "unique_destinations": unique_dest,
            "failed_auth_count": failed_auth,
            "label": label,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data):,} synthetic rows to: {OUTPUT}")
    print("Reminder: this is NOT official AWID3 and is not real captured traffic.")
