import argparse
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import shutil
import time
from pathlib import Path

# ================= CONFIGURATION =================
# 1. CHANGE THIS to your actual Google Drive path
DRIVE_ROOT = Path("G:/My Drive/ML/qari")

# 2. Local Paths
LOCAL_RESEARCH = Path(__file__).resolve().parent.parent
LOCAL_PROJECT_ROOT = LOCAL_RESEARCH.parent

# 3. PUSH IGNORE (What NOT to upload to Drive)
# We exclude heavy artifacts that Colab will regenerate or that are useless there.
PUSH_IGNORE = shutil.ignore_patterns(
    "venv", "venv_*", ".git", "__pycache__", ".idea", "node_modules", "wandb",
    "*.h5",  # Don't push local heavy models (we pull them instead)
    "features.npz",  # Don't push features (regenerate on Colab is faster)
    "*.zip", "*.7z",  # Don't push archives
    "miccap*", "raw_mic*",  # Don't push local debug recordings
    "tfjs_model",  # Don't push old web models
    ".DS_Store",
    "sound_datasets",
    "archive"
)

# 4. PULL IGNORE (What NOT to overwrite locally)
# When pulling, we want everything usually, but we can exclude system files
PULL_IGNORE = shutil.ignore_patterns(
    "venv", "__pycache__", ".DS_Store"
)


# =================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def ensure_drive_connected():
    if not DRIVE_ROOT.parent.exists():
        log(f"❌ Error: Google Drive path not found at: {DRIVE_ROOT.parent}")
        log("   Please edit DRIVE_ROOT in 'tools/sync_manager.py'")
        exit(1)


def _changed(src_file: Path, dst_file: Path) -> bool:
    """
    Returns True if src_file should be copied to dst_file.

    Heuristic (fast):
    - copy if destination missing
    - copy if file size differs
    - copy if mtime differs (rounded to int to avoid tiny FS resolution issues)
    """
    if not dst_file.exists():
        return True

    try:
        s = src_file.stat()
        d = dst_file.stat()
    except OSError:
        return True

    if s.st_size != d.st_size:
        return True

    if int(s.st_mtime) != int(d.st_mtime):
        return True

    return False


def sync_folder_incremental(src: Path, dst: Path, ignore_patterns):
    """
    Incremental Sync (PUSH only):
    Walks src and copies ONLY files that changed (new/modified),
    skipping ignored paths. Does NOT delete anything in dst.
    """
    if not src.exists():
        log(f"⚠️ Source missing: {src}")
        return

    log(f"📂 Syncing (incremental): {src.name}/  ->  {dst}")

    copied = 0
    skipped = 0

    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)

        # Apply ignore patterns at this directory level
        names = list(dirs) + list(files)
        ignored = set()
        if ignore_patterns:
            ignored = set(ignore_patterns(str(root_p), names))

        # Prevent descending into ignored directories
        dirs[:] = [d for d in dirs if d not in ignored]

        for fn in files:
            if fn in ignored:
                continue

            sfile = root_p / fn
            dfile = dst_root / fn

            if _changed(sfile, dfile):
                dfile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sfile, dfile)  # preserves mtime/metadata
                copied += 1
            else:
                skipped += 1

    log(f"   ✅ Incremental sync complete. Copied: {copied}, Skipped: {skipped}")


def sync_folder_copytree(src: Path, dst: Path, ignore_patterns):
    """
    Basic Sync (used for PULL):
    Copies src to dst, skipping ignored files.
    """
    if not src.exists():
        log(f"⚠️ Source missing: {src}")
        return

    log(f"📂 Syncing: {src.name}/  ->  {dst}")

    try:
        # copytree with dirs_exist_ok=True acts like a "Merge/Update"
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_patterns)
    except Exception as e:
        log(f"   ❌ Error syncing {src.name}: {e}")


def push_to_drive():
    """Local -> Google Drive (Code + Raw Audio)"""
    ensure_drive_connected()
    log("🚀 STARTING PUSH (Local -> Drive)...")

    # 1. Push Research (Code, Raw Audio in datasets/audio, Configs)
    src = LOCAL_RESEARCH
    dst = DRIVE_ROOT / "research"
    sync_folder_incremental(src, dst, PUSH_IGNORE)

    # 2. Push App Configs (reciters_map.json, audio_config.json)
    # These are small and critical for Colab to know what to train
    src = LOCAL_PROJECT_ROOT / "app" / "public" / "models"
    dst = DRIVE_ROOT / "app" / "public" / "models"

    # We explicitly allow json here, but exclude binary weights if any exist locally
    APP_IGNORE = shutil.ignore_patterns("*.bin", "*.h5")
    sync_folder_incremental(src, dst, APP_IGNORE)

    log("✅ Push Complete. Code and Raw Data are on Drive.")


def pull_from_drive():
    """Google Drive -> Local (Trained Models Only)"""
    ensure_drive_connected()
    log("📥 STARTING PULL (Drive -> Local)...")

    # 1. Pull Trained H5 Models
    src_models = DRIVE_ROOT / "research" / "models"
    dst_models = LOCAL_RESEARCH / "models"

    if src_models.exists():
        log("📦 Pulling updated .h5 models...")
        # We only want .h5 and .json (baselines) from here
        MODEL_ONLY_IGNORE = shutil.ignore_patterns("*.py", "*.txt")
        sync_folder_copytree(src_models, dst_models, MODEL_ONLY_IGNORE)
    else:
        log("⚠️ No models folder found in Drive yet.")

    # 2. Pull TFJS Web Files (The "Wizard" Output)
    # This includes: model.json, group1-shard1of1.bin, etc.
    src_web = DRIVE_ROOT / "app" / "public" / "models"
    dst_web = LOCAL_PROJECT_ROOT / "app" / "public" / "models"

    if src_web.exists():
        log("🌐 Pulling TFJS web model files...")
        sync_folder_copytree(src_web, dst_web, PULL_IGNORE)
    else:
        log("⚠️ No web models found in Drive yet.")

    log("✅ Pull Complete. Your local app has the latest brain!")


def main():
    p = argparse.ArgumentParser(description="Sync Manager for Qari Finder")
    p.add_argument("--push", action="store_true", help="Upload Code & Data to Drive")
    p.add_argument("--pull", action="store_true", help="Download Trained Models from Drive")
    args = p.parse_args()

    if args.push:
        push_to_drive()
    elif args.pull:
        pull_from_drive()
    else:
        print("Please specify --push or --pull")


if __name__ == "__main__":
    main()
