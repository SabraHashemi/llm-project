"""
Quick sync script to push local changes to GitHub for Colab use.

Usage:
    python sync_to_colab.py [commit_message]

This script:
1. Stages all changes
2. Commits with message
3. Pushes to GitHub
4. Prints instructions for Colab
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a shell command"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return None

def sync_to_colab(commit_message="Auto-sync to Colab"):
    """Sync local changes to GitHub"""
    
    print("🔄 Syncing to GitHub for Colab use...")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("❌ Git not initialized. Run: git init")
        return False
    
    # Check if remote exists
    result = run_command('git remote -v', check=False)
    if not result:
        print("❌ No GitHub remote found.")
        print("   Set it up with: git remote add origin YOUR_REPO_URL")
        return False
    
    # Stage all changes
    print("📦 Staging changes...")
    run_command('git add .')
    
    # Check if there are changes
    status = run_command('git status --porcelain')
    if not status:
        print("✅ No changes to commit!")
        return True
    
    # Commit
    print(f"💾 Committing: {commit_message}")
    result = run_command(f'git commit -m "{commit_message}"')
    if not result:
        print("⚠️  No changes to commit (or commit failed)")
        return False
    
    # Push
    print("🚀 Pushing to GitHub...")
    result = run_command('git push')
    if not result:
        print("❌ Push failed. Check your GitHub credentials.")
        return False
    
    print("\n✅ Successfully synced to GitHub!")
    print("\n📋 Next steps in Colab:")
    print("   !git pull")
    print("   # or")
    print("   !git clone https://github.com/SabraHashemi/llm-project.git")
    print("   %cd llm-project")
    
    return True

if __name__ == "__main__":
    commit_msg = sys.argv[1] if len(sys.argv) > 1 else "Auto-sync to Colab"
    sync_to_colab(commit_msg)

