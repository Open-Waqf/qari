import asyncio
import json
import os
import socket
import subprocess
import sys

import numpy as np
from playwright.async_api import async_playwright

# ================= CONFIGURATION =================
# We no longer hardcode APP_URL. We auto-discover it.
AUDIO_CONFIG_PATH = "../app/public/models/audio_config.json"
TOLERANCE_THRESHOLD = 0.05


# =================================================

def python_compute_mfcc(signal, config):
    """
    Python Ground Truth implementation.
    Must match CustomAudioExtractor logic exactly.
    """
    # 1. Load Matrices
    dft_real = np.array(config["dft_real"])
    dft_imag = np.array(config["dft_imag"])
    mel_basis = np.array(config["mel_basis"])
    dct_matrix = np.array(config["dct_matrix"])
    window = np.array(config["window"])

    # 2. Frame Setup
    # Pad or cut to exactly 512 samples
    if len(signal) != 512:
        new_sig = np.zeros(512, dtype=np.float32)
        length = min(len(signal), 512)
        new_sig[:length] = signal[:length]
        signal = new_sig

    # 3. Windowing
    windowed = signal * window

    # 4. DFT
    real = dft_real @ windowed
    imag = dft_imag @ windowed
    magnitude = np.sqrt(real ** 2 + imag ** 2)

    # 5. Mel Spectrogram
    mel_energies = mel_basis @ magnitude

    # 6. Log
    log_mel = np.log(mel_energies + 1e-6)

    # 7. MFCC (DCT)
    mfccs = dct_matrix @ log_mel

    return mfccs


async def find_running_app(browser):
    """
    Auto-detects the URL where the app is serving.
    Tries Localhost, Hostname, and WSL Gateway.
    """
    candidates = [
        "http://localhost:5173/?debug=1",
        "http://127.0.0.1:5173/?debug=1"
    ]

    # 1. Try Computer Name
    try:
        hostname = socket.gethostname()
        if hostname:
            candidates.append(f"http://{hostname}:5173/?debug=1")
    except:
        pass

    # 2. Try WSL Gateway IP (Linux -> Windows Bridge)
    try:
        if sys.platform.startswith("linux"):
            cmd = "ip route show | grep default | awk '{print $3}'"
            host_ip = subprocess.check_output(cmd, shell=True).decode().strip()
            if host_ip:
                candidates.append(f"http://{host_ip}:5173/?debug=1")
                print(f"   (Detected WSL Bridge IP: {host_ip})")
    except:
        pass

    print(f"🔍 Scanning for App on: {candidates}...")

    for url in candidates:
        try:
            context = await browser.new_context()
            page = await context.new_page()
            # Fast timeout (2s) to fail quickly
            response = await page.goto(url, timeout=2000, wait_until='domcontentloaded')
            await context.close()

            # If we get here without error, it connected!
            print(f"✅ Found App at: {url}")
            return url
        except Exception:
            # print(f"   ❌ No response from {url}")
            continue

    return None


async def run_parity_check():
    # 1. Load Config
    if not os.path.exists(AUDIO_CONFIG_PATH):
        # Handle running from root vs research folder
        alt_path = "app/public/models/audio_config.json"
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"❌ Error: Config not found at {AUDIO_CONFIG_PATH}")
            sys.exit(1)
    else:
        config_path = AUDIO_CONFIG_PATH

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Generate Signal
    t = np.linspace(0, 1, 512)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
              0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    # 3. Python Math
    print("🐍 Computing Python Baseline...")
    py_result = python_compute_mfcc(signal, config)

    # 4. App Math (Playwright)
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)

            # --- AUTO DISCOVERY START ---
            app_url = await find_running_app(browser)

            if not app_url:
                print("\n❌ Error: Could not find running app.")
                print("   👉 Is 'npm run dev' running?")
                print("   👉 Did you add '--host' to package.json? (e.g. 'vite --host')")
                await browser.close()
                sys.exit(1)
            # ----------------------------

            page = await browser.new_page()
            await page.goto(app_url)
            await page.wait_for_timeout(1000)

            # Check Hook
            is_hooked = await page.evaluate("typeof window.checkParity === 'function'")
            if not is_hooked:
                print("❌ Error: 'window.checkParity' not found.")
                print("   👉 Ensure 'debug=1' logic is active in main.ts")
                await browser.close()
                sys.exit(1)

            print("🌐 Sending Signal to App...")

            signal_list = signal.tolist()
            js_result_list = await page.evaluate(f"window.checkParity({signal_list})")

            if not js_result_list:
                print("❌ Error: App returned empty result.")
                await browser.close()
                sys.exit(1)

            js_result = np.array(js_result_list).flatten()
            await browser.close()

        except Exception as e:
            print(f"❌ Playwright Error: {e}")
            sys.exit(1)

    # 5. COMPARE
    print("\n📊 --- PARITY REPORT ---")

    if py_result.shape != js_result.shape:
        # Try to align shapes if flattened vs 2D
        py_result = py_result.flatten()
        js_result = js_result.flatten()

    diff = np.abs(py_result - js_result)
    max_diff = np.max(diff)

    print(f"Max Drift:     {max_diff:.8f}")

    print("\n--- Sample ---")
    print(f"Python: {py_result[:3]} ...")
    print(f"App:    {js_result[:3]} ...")

    if max_diff > TOLERANCE_THRESHOLD:
        print(f"\n❌ FAILED: Math Mismatch (> {TOLERANCE_THRESHOLD})")
        sys.exit(1)
    else:
        print("\n✅ PASSED: Physics Engine is Identical.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(run_parity_check())
