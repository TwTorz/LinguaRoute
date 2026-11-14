# Git Cheat Sheet for Teams

## Daily commands
```bash
git pull                     # get teammates' latest changes (do this first!)
git status                   # see what changed
git add .                    # stage all changes
git commit -m "Add login page"
git push                     # upload to GitHub
```

## Create the GitHub repo (once, from this folder)
```bash
gh repo create my-hackathon-project --public --source=. --push
```

## Add teammates
```bash
gh repo edit --add-collaborator <their-github-username>   # or: GitHub → Settings → Collaborators
```
Teammates then run:
```bash
gh repo clone <your-username>/my-hackathon-project
```

## Working on separate features (fewer conflicts)
```bash
git checkout -b feature-name   # make a branch
# ...work, add, commit...
git push -u origin feature-name
gh pr create --fill            # open a pull request
gh pr merge --merge            # merge it when it's ready
git checkout main && git pull
```

## Fixing a merge conflict
1. `git pull` shows CONFLICT in some file
2. Open the file and find the `<<<<<<<`, `=======`, `>>>>>>>` markers
3. Keep the right code and delete the markers
4. `git add <file>` → `git commit` → `git push`

## Oops
```bash
git restore <file>             # throw away uncommitted changes to a file
git commit --amend -m "msg"    # fix the last commit message (before pushing)
git log --oneline              # see history
```

## Tips
- Never commit API keys; put them in `.env` (already ignored)
- Pull before you start, push when you stop
- Small, frequent commits are better than one giant one
