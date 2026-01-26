#!/usr/bin/env python3
"""
Build Script - Convenience script to build and run the RAG system.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """Main build script."""
    print("\n" + "="*60)
    print("  HYBRID RAG SYSTEM - BUILD SCRIPT")
    print("="*60)

    actions = {
        'docker': 'Build and run with Docker',
        'local': 'Run locally (requires dependencies)',
        'index': 'Build indices only',
        'evaluate': 'Run evaluation',
        'ui': 'Start Streamlit UI'
    }

    if len(sys.argv) < 2:
        print("\nUsage: python build.py <action>")
        print("\nAvailable actions:")
        for action, desc in actions.items():
            print(f"  {action:12} - {desc}")
        return

    action = sys.argv[1]

    if action == 'docker':
        run_command('docker-compose up --build', 'Building Docker containers')

    elif action == 'local':
        run_command('pip install -r requirements.txt', 'Installing dependencies')
        run_command('streamlit run app.py', 'Starting Streamlit UI')

    elif action == 'index':
        run_command('python main.py --build-index --generate-questions', 'Building indices')

    elif action == 'evaluate':
        run_command('python main.py --evaluate', 'Running evaluation')

    elif action == 'ui':
        run_command('streamlit run app.py', 'Starting UI')

    else:
        print(f"Unknown action: {action}")
        print(f"Available: {', '.join(actions.keys())}")


if __name__ == "__main__":
    main()
