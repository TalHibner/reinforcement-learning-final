# =========================================================
# Single source of truth for where artifacts are persisted.
# Priority: mounted Google Drive -> Colab session storage -> local folder.
# Nothing later in the notebook redefines STORAGE_ROOT, so a local rerun
# (no /content) writes next to the notebook instead of crashing.
# =========================================================
DRIVE_ROOT = "/content/drive/MyDrive/rl_final_project"
COLAB_ROOT = "/content/rl_final_project"
LOCAL_ROOT = os.path.join(os.path.abspath(os.getcwd()), "rl_final_project")

if os.path.isdir("/content/drive/MyDrive"):
    STORAGE_ROOT = DRIVE_ROOT      # Colab + Drive: survives session end
elif os.path.isdir("/content"):
    STORAGE_ROOT = COLAB_ROOT      # Colab, no Drive: wiped at session end
else:
    STORAGE_ROOT = LOCAL_ROOT      # local Jupyter / grader rerun

os.makedirs(STORAGE_ROOT, exist_ok=True)

# Everything that persists lives under STORAGE_ROOT.
ARTIFACTS = os.path.join(STORAGE_ROOT, "artifacts")
for sub in ("logs", "plots", "checkpoints"):
    os.makedirs(os.path.join(ARTIFACTS, sub), exist_ok=True)

print(f"STORAGE_ROOT: {STORAGE_ROOT}")
print(f"ARTIFACTS:    {ARTIFACTS}")
