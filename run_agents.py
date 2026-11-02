import subprocess
import sys
import time

ROLES = ["proponent", "opponent", "evaluator"]

def main():
    processes = []
    print("Starting all debate agents...")

    try:
        for role in ROLES:
            print(f"  -> Starting {role} agent server...")
            proc = subprocess.Popen([sys.executable, "agent_server.py", role])
            processes.append(proc)

        print("\nAll 3 agent servers are running (Proponent :8001, Opponent :8002, Evaluator :8003).")
        print("Now open another terminal and run:")
        print("  python coordinator.py\n")
        print("Press Ctrl+C to terminate all agent servers.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down all agent servers...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("All servers stopped.")

if __name__ == "__main__":
    main()

