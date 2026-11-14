# Git Cheat Sheet

## Create the GitHub repo (once, from this folder)
```bash
gh repo create my-hackathon-project --public --source=. --push
```

## Save your work (do this often)
```bash
git add .                        # stage all changes
git commit -m "Add login page"   # save a checkpoint
git push                         # upload to GitHub
```

## Trying something risky?
Make a branch so you can throw it away if it doesn't work:
```bash
git checkout -b experiment       # try stuff here
git checkout main                # go back to the safe version
git merge experiment             # keep it if it worked
```

## Oops
```bash
git status                       # what's changed?
git restore <file>               # undo uncommitted changes to a file
git restore .                    # undo ALL uncommitted changes (careful!)
git log --oneline                # see your checkpoints
git checkout <commit-id> -- <file>   # bring back an old version of a file
```

## Tips
- Never commit API keys; put them in `.env` (already ignored)
- Commit every time something works: free undo points
- Judges may look at your commit history, so steady commits show you built it during the event
