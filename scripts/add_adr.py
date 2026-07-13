import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

# Setup logger following guidelines
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("add_adr")

def main() -> None:
    parser = argparse.ArgumentParser(description="Add an ADR entry automatically and run archiver.")
    parser.add_argument("--task", required=True, help="Task ID (e.g. TASK_L0_MTF_FUSION_FACTORY)")
    parser.add_argument("--title", required=True, help="Title of the decision")
    parser.add_argument("--why", required=True, help="Context or why this decision was made")
    parser.add_argument("--what", required=True, help="Resolution or what was implemented")
    parser.add_argument("--impact", required=True, help="Impact of the change")
    
    args = parser.parse_args()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    adr_date = datetime.now().strftime("%Y%m%d")
    adr_id = f"ADR_{adr_date}_{args.task.replace('TASK_', '')}"
    
    decisions_path = "docs/decisions/decisions.md"
    if not os.path.exists(decisions_path):
        logger.error(f"Decisions log file not found at: {decisions_path}")
        sys.exit(1)
        
    with open(decisions_path, encoding="utf-8") as f:
        content = f.read()
        
    # Standard format entry (max 5 lines total)
    new_entry = (
        f"## [{date_str}] [{args.task}] [{adr_id}]\n"
        f"- **Context/Why:** {args.why}\n"
        f"- **Resolution/What:** {args.what}\n"
        f"- **Impact:** {args.impact}\n\n"
    )
    
    # Insert right below the title header `# Active Decisions Log (Sliding Window)`
    header_marker = "# Active Decisions Log (Sliding Window)\n"
    if header_marker in content:
        parts = content.split(header_marker, 1)
        # Handle optional spacing below header
        after_header = parts[1].lstrip()
        updated_content = f"{header_marker}\n{new_entry}{after_header}"
    else:
        # Fallback to appending at the top
        logger.warning("Header marker not found, prepending to file.")
        updated_content = f"{new_entry}{content}"
        
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
        
    logger.info("Successfully added new ADR entry.")
    
    # Automatically run archive script to maintain the 15-entry limit
    archive_cmd = ["uv", "run", "python", "scripts/archive_decisions.py", "--max-entries", "15"]
    logger.info(f"Running archive command: {' '.join(archive_cmd)}")
    res = subprocess.run(archive_cmd, capture_output=True, text=True, shell=False)  # noqa: S603
    if res.returncode == 0:
        logger.info("Archive script completed successfully.")
    else:
        logger.error(f"Archive script failed: {res.stderr}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
