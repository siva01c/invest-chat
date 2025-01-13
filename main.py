import subprocess
import os
import threading
import argparse
from src.pdf_extractor import process_pdf_directory

def run_flask():
    script_path = os.path.join(os.path.dirname(__file__), 'src', 'webserver.py')
    subprocess.run(['python', script_path])

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    parser = argparse.ArgumentParser(description='Run PDF processor and Flask server')
    parser.add_argument('--run-extractor', action='store_true', help='Run the PDF extractor')
    args = parser.parse_args()

    if args.run_extractor:
        process_pdf_directory("datasources")
    
    flask_thread.start()
    flask_thread.join()
