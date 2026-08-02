# Releasing to Public Repository

The music video pipeline lives in the private Alice repository at `scripts/mv-public/`. Changes are pushed to the public `alice-music-video` repository via git subtree.

## Prerequisites

- Git subtree configured in the private Alice repo
- Remote `mv-public` pointing to the alice-music-video repository

## Push Changes

```bash
cd ~/alice
git subtree push --prefix=scripts/mv-public mv-public main
```

This pushes the contents of `scripts/mv-public/` to the `main` branch of the `mv-public` remote in a single command.

## Split and Push (Manual)

If you need more control (e.g., reviewing the split before pushing):

```bash
cd ~/alice
git subtree split --prefix=scripts/mv-public -b mv-public-branch
git push mv-public mv-public-branch:main
git branch -D mv-public-branch
```

## Verify

Confirm the public repository received the changes:

```bash
git log mv-public/main --oneline -5
```

## Notes

- The subtree approach preserves the full git history of the mv-public directory.
- Only files under `scripts/mv-public/` are included in the public repository.
- Sensitive files (`.env`, project data, audio/video media) are excluded via `.gitignore`.
