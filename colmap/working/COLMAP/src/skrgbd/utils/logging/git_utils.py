import subprocess


def get_git_revision(repo_dir):
    return subprocess.check_output('git rev-parse HEAD'.split(), cwd=repo_dir, text=True)[:-1]
