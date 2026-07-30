# Drive is mounted (if available) in the Installs cell above, and STORAGE_ROOT is
# defined there — this cell only adds the display/video helpers that build on it.


def video_path(name: str, folder_name: str = "videos") -> str:
    """Return a writable absolute path for a video file under ``folder_name/``.

    Videos are placed under STORAGE_ROOT so they survive Colab session end when
    STORAGE_ROOT points to Google Drive, and land in a local folder otherwise.
    """
    folder = os.path.join(STORAGE_ROOT, folder_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


print("STORAGE_ROOT:", STORAGE_ROOT)
print("Drive mounted?", os.path.isdir("/content/drive/MyDrive"))
print("ARTIFACTS exists?", os.path.isdir(ARTIFACTS))
print("Sample video path:", video_path("sanity_test.mp4"))


def embed_mp4(filename: str):
    """Embed an mp4 file inline. Works in Colab and local Jupyter alike."""
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    tag = (
        '<video width="640" height="480" controls>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
        "Your browser does not support the video tag."
        "</video>"
    )
    return HTML(tag)


def random_rollout_video(env, filename, max_steps=100, fps=10, seed=None):
    """Run one random-action episode and save it as mp4. Returns (steps, total_reward)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    with imageio.get_writer(filename, fps=fps) as video:
        video.append_data(env.render())
        for steps in range(1, max_steps + 1):
            action = random.randint(0, env.action_space.n - 1)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            video.append_data(env.render())
            if terminated or truncated:
                break
    return steps, total_reward
