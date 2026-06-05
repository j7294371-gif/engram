#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Push memore to GitHub — one script to publish the project
# ─────────────────────────────────────────────────────────────
# Usage:
#   1.  cd scripts/ && bash push_to_github.sh
#   2.  Follow the browser login prompt (if needed)
#   3.  Done — your project is live at github.com/j7294371-gif/memore
# ─────────────────────────────────────────────────────────────

set -e

REPO_NAME="memore"
GITHUB_USER="j7294371"
REMOTE_URL="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"

echo "============================================"
echo " 🚀  Publishing memore to GitHub"
echo "============================================"
echo ""

# Step 1: Make sure we're in the right directory
cd "$(dirname "$0")/.."
echo "📁  Project root: $(pwd)"

# Step 2: Check if gh CLI is available
GH=""
if command -v gh &>/dev/null; then
    GH="gh"
elif [ -f "/c/Users/26525/AppData/Local/Temp/gh_cli/bin/gh.exe" ]; then
    GH="/c/Users/26525/AppData/Local/Temp/gh_cli/bin/gh.exe"
else
    echo "⚠️   GitHub CLI not found. Installing..."
    echo "   Download from: https://cli.github.com/"
    echo "   Or run:  winget install --id GitHub.cli"
    echo ""
    echo "   After installing, re-run this script."
    exit 1
fi

echo "🔑  GitHub CLI: $($GH --version | head -1)"

# Step 3: Authenticate with GitHub
echo ""
echo "🔐  Logging in to GitHub..."
echo "    (Follow the browser prompt to authorize)"
echo ""
$GH auth login --git-protocol https -h github.com -s repo,workflow 2>&1 || {
    echo ""
    echo "❌  Auth failed. Let's try SSH..."
    echo ""

    # Generate SSH key if needed
    if [ ! -f ~/.ssh/id_ed25519 ]; then
        echo "   Generating SSH key..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$(git config user.email)"
    fi

    echo "   Your public SSH key:"
    cat ~/.ssh/id_ed25519.pub
    echo ""
    echo "   Please add this key to GitHub:"
    echo "   https://github.com/settings/keys"
    echo ""
    read -p "   Press Enter after adding the key... " -r

    # Try SSH auth
    $GH auth login --git-protocol ssh -h github.com -s repo,workflow 2>&1
}

echo ""
echo "✅  Authenticated!"

# Step 4: Create the GitHub repository
echo ""
echo "📦  Creating GitHub repository..."
$GH repo create ${REPO_NAME} --public --description "Biomimetic memory system for AI agents — sensory, working, and long-term memory with forgetting curves, sleep consolidation, and associative retrieval." --source=. --push 2>&1 || {
    # If repo exists, just add remote and push
    echo "   Repository may already exist. Setting up remote..."
    git remote remove origin 2>/dev/null || true
    git remote add origin "${REMOTE_URL}"
    git branch -M main
    git push -u origin main 2>&1
}

echo ""
echo "============================================"
echo " ✅  Done! Your project is live at:"
echo "     https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo " 📦  Install from GitHub:"
echo "     pip install git+https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
echo ""
echo " 📦  Or after PyPI release:"
echo "     pip install memore"
echo "============================================"
