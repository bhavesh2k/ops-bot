import subprocess
import sys

def run_command(command, is_module=False):
    """
    Executes a python command. 
    If is_module is True, it runs 'python -m path.to.module'
    """
    prefix = [sys.executable]
    if is_module:
        prefix.append("-m")
    
    full_cmd = prefix + [command]
    
    print(f"🚀 Running: {' '.join(full_cmd)}")
    
    try:
        subprocess.run(full_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Step '{command}' failed.")
        sys.exit(1)

if __name__ == "__main__":
    # 1. Build Chunks (Direct script run)
    run_command("scripts/build_chunks.py")

    # 2. Embed Chunks (Direct script run)
    run_command("vector/embed_chunks_local.py")

    # 3. Generate Answer (Module run to handle relative imports)
    # We convert the path 'assistant/generate_answer.py' to 'assistant.generate_answer'
    run_command("assistant.generate_answer", is_module=True)

    print("\n✨ Pipeline completed successfully!")