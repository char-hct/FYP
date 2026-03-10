import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Data collection scripts (run in parallel or sequence)
data_collection_scripts = [
    'macro_data.py',
    'more_market.py',
    'more_news.py',
    'moree.py',
]

# Step 2: Data processing scripts (run in sequence, after collection)
data_processing_scripts = [
    'checking.py',
    'fill_nan.py',
    's_trans.py',
]

def run_scripts(scripts, stop_on_error=True):
    failed_scripts = []
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), '..', script)
        script_path = os.path.abspath(script_path)
        if os.path.exists(script_path):
            print(f"Running {script}...")
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f"[FAIL] {script} failed:\n{result.stderr}")
                failed_scripts.append(script)
                if stop_on_error:
                    break
        else:
            print(f"Script not found: {script}")
            failed_scripts.append(script)
            if stop_on_error:
                break
    return failed_scripts

if __name__ == "__main__":
    print("=== Step 1: Data Collection ===")
    failed_data_scripts = run_scripts(data_collection_scripts, stop_on_error=False)
    if failed_data_scripts:
        print("\n[WARNING] The following data collection scripts failed:")
        for script in failed_data_scripts:
            print(f" - {script}")
    else:
        print("All data collection scripts succeeded.")

    # Check if more_news.py fetched new data
    more_news_log = "more_news_new_count.log"
    news_script = os.path.join(os.path.dirname(__file__), '..', 'more_news.py')
    import re
    new_news_count = None
    if os.path.exists(news_script):
        # Patch more_news.py to write new count to a log file
        # (Assume the last print line is: print(f"Done! {len(df_new)} new articles saved. Total: {len(df_combined)} articles.") )
        # So we can parse this from stdout
        result = subprocess.run([sys.executable, news_script], capture_output=True, text=True)
        if "NO_NEW_NEWS" in result.stdout:
            new_news_count = 0
        else:
            match = re.search(r"Done! (\\d+) new articles saved", result.stdout)
            if match:
                new_news_count = int(match.group(1))
    else:
        print("Warning: more_news.py not found for new data check.")

    # If new_news_count is not None and > 0, run the sentiment scripts
    if new_news_count is not None and new_news_count > 0:
        print(f"New news data found: {new_news_count} new articles. Running sentiment scripts...")
        sentiment_scripts = [
            'news_sentiment_model.py',
            'news_sentiment_ai.py',
            'news_trans.py',
        ]
        failed_processing_scripts = run_scripts(sentiment_scripts + data_processing_scripts, stop_on_error=True)
        if failed_processing_scripts:
            print("\n[ERROR] The following data processing script failed and stopped the pipeline:")
            for script in failed_processing_scripts:
                print(f" - {script}")
        else:
            print("All data processing scripts succeeded.")
    else:
        print("No new news data found. Skipping sentiment scripts.")
        # Only run the remaining data processing scripts, not any sentiment or translation scripts
        failed_processing_scripts = run_scripts(data_processing_scripts, stop_on_error=True)
        if failed_processing_scripts:
            print("\n[ERROR] The following data processing script failed and stopped the pipeline:")
            for script in failed_processing_scripts:
                print(f" - {script}")
        else:
            print("All data processing scripts succeeded.")

    # Final step: Extract model predictions for trading simulation
    print("\n=== Step 3: Extract Model Predictions ===")
    prediction_extraction_script = 'extract_model_predictions.py'
    extraction_script_path = os.path.join(os.path.dirname(__file__), '..', prediction_extraction_script)
    extraction_script_path = os.path.abspath(extraction_script_path)
    
    if os.path.exists(extraction_script_path):
        print(f"Running {prediction_extraction_script}...")
        result = subprocess.run([sys.executable, extraction_script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[WARNING] Model prediction extraction had issues:\n{result.stderr}")
        else:
            print("✓ Model predictions extracted successfully!")
    else:
        print(f"Note: {prediction_extraction_script} not found. Skipping prediction extraction.")

    # Step 4: Generate AI weekly news summary
    print("\n=== Step 4: Generate Weekly News Summary ===")
    summary_script = 'generate_weekly_summary.py'
    summary_script_path = os.path.join(os.path.dirname(__file__), '..', summary_script)
    summary_script_path = os.path.abspath(summary_script_path)

    if os.path.exists(summary_script_path):
        print(f"Running {summary_script}...")
        result = subprocess.run([sys.executable, summary_script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[WARNING] Weekly summary generation had issues:\n{result.stderr}")
        else:
            print("✓ Weekly news summary generated successfully!")
    else:
        print(f"Note: {summary_script} not found. Skipping summary generation.")
    
    print("\n=== Pipeline Complete ===")
    print("All update steps finished.")
