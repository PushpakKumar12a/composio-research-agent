import argparse
import sys

from src.research_agent import run_research, run_test
from src.verification_agent import run_verification
from src.html_generator import generate_html_report
from config.settings import parse_apps_file

def main():
    parser = argparse.ArgumentParser(description="Run the full research pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 apps only")
    parser.add_argument("--skip-research", action="store_true", help="Skip research, regenerate HTML")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted research")
    args = parser.parse_args()

    print("Composio 100 App Research Pipeline")
    print("AI Product Ops - Toolkit Buildability Analysis")

    if not args.skip_research:
        print("\nStep 1: Running Research Agent...")
        if args.test:
            run_test()
        else:
            apps = parse_apps_file()
            run_research(apps, resume=args.resume)
    else:
        print("\nSkipping research (using existing data)")

    if not args.skip_verify and not args.test:
        print("\nStep 2: Running Verification Agent...")
        run_verification(sample_size=20)
    else:
        print("\nSkipping verification")

    print("\nStep 3: Generating HTML Deliverable...")
    generate_html_report()

    print("\nPipeline Complete!")
    print("Open report.html in your browser to view the report.")

if __name__ == "__main__":
    main()
