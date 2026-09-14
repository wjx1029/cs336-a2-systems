import subprocess
import os


model_sizes = [
    ("small", 768, 3072, 12, 12),
    # ("medium", 1024, 4096, 24, 16),
    # ("large", 1280, 5120, 36, 20),
    # ("xl", 2560, 10240, 32, 32),
    # ("10B", 4608, 12288, 50, 36),
]

for name, d_model, d_ff, num_layers, num_heads in model_sizes:
    print(f"\n===== Running model: {name} =====")
    for mode in ['forward-only', 'forward-backward', 'forward-optimizer']:
        cmd = [
            "uv", "run", os.path.abspath(os.path.join(os.path.dirname(__file__), 'benchmark.py')),
            "--d_model", str(d_model),
            "--d_ff", str(d_ff),
            "--num_layers", str(num_layers),
            "--num_heads", str(num_heads),
            "--vocab_size", "10000",
            "--context_length", "512",
            "--batch_size", "4",
            "--warmup", "5",
            "--m_steps", "10",
            "--mode", mode,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("stdout:\n", res.stdout)
        if res.stderr:
            print("stderr:\n", res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"({name},{mode}) failed.")