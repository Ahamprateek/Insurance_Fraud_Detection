import sqlite3
from datetime import datetime
import os
import re
import json
import shutil
import pandas as pd
import logging
from pathlib import Path


class Raw_Data_validation:
    def __init__(self, batch_directory):
        self.batch_directory = Path(batch_directory)
        self.schema_path = Path('schema_training.json')
        self.logger = self._setup_logger()
        self.validation_results = {"good_files": [], "bad_files": []}

    def _setup_logger(self):
        """Setup structured logging"""
        logger = logging.getLogger('DataValidation')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler('Training_Logs/validation.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def values_from_schema(self):
        """Extract schema validation parameters with robust error handling"""
        try:
            if not self.schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

            with open(self.schema_path, 'r') as f:
                schema = json.load(f)

            date_len = schema['LengthOfDateStampInFile']
            time_len = schema['LengthOfTimeStampInFile']
            col_names = schema['ColName']
            num_cols = schema['NumberofColumns']

            self.logger.info(f"Schema loaded: date_len={date_len}, time_len={time_len}, cols={num_cols}")
            return date_len, time_len, col_names, num_cols

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Schema parsing error: {e}")
            raise ValueError(f"Invalid schema: {e}")

    def create_filename_regex(self, date_len, time_len):
        """Create precise regex for fraudDetection files"""
        return re.compile(rf"fraudDetection_\d{{{date_len}}}\d{{{time_len}}}\.csv$")

    def create_directories(self):
        """Create Good_Raw and Bad_Raw directories safely"""
        good_dir = Path("Training_Raw_files_validated/Good_Raw")
        bad_dir = Path("Training_Raw_files_validated/Bad_Raw")

        good_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Directories created")

    def clear_directories(self):
        """Safely clear existing directories"""
        good_dir = Path("Training_Raw_files_validated/Good_Raw")
        bad_dir = Path("Training_Raw_files_validated/Bad_Raw")

        for dir_path in [good_dir, bad_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
        self.logger.info("Existing directories cleared")

    def validate_filenames(self, date_len, time_len):
        """Validate filename format and move files"""
        self.clear_directories()
        self.create_directories()

        regex = self.create_filename_regex(date_len, time_len)
        csv_files = list(self.batch_directory.glob("*.csv"))

        self.logger.info(f"Found {len(csv_files)} CSV files for validation")

        for file_path in csv_files:
            filename = file_path.name
            if regex.match(filename):
                parts = filename.replace('.csv', '').split('_')
                if (len(parts) == 3 and
                        len(parts[1]) == date_len and
                        len(parts[2]) == time_len):

                    dest = Path("Training_Raw_files_validated/Good_Raw") / filename
                    shutil.copy2(file_path, dest)
                    self.validation_results["good_files"].append(filename)
                    self.logger.info(f"✓ Valid: {filename}")
                else:
                    dest = Path("Training_Raw_files_validated/Bad_Raw") / filename
                    shutil.copy2(file_path, dest)
                    self.validation_results["bad_files"].append(filename)
                    self.logger.error(f"✗ Invalid format: {filename}")
            else:
                dest = Path("Training_Raw_files_validated/Bad_Raw") / filename
                shutil.copy2(file_path, dest)
                self.validation_results["bad_files"].append(filename)
                self.logger.error(f"✗ Regex fail: {filename}")

    def validate_column_count(self, expected_cols):
        """Validate number of columns in good files"""
        good_dir = Path("Training_Raw_files_validated/Good_Raw")
        good_files = list(good_dir.glob("*.csv"))

        for file_path in good_files:
            try:
                df = pd.read_csv(file_path)
                if df.shape[1] != expected_cols:
                    bad_dest = Path("Training_Raw_files_validated/Bad_Raw") / file_path.name
                    shutil.move(str(file_path), str(bad_dest))
                    self.validation_results["bad_files"].append(file_path.name)
                    self.validation_results["good_files"].remove(file_path.name)
                    self.logger.warning(f"✗ Column count {df.shape[1]} != {expected_cols}: {file_path.name}")
                else:
                    self.logger.info(f"✓ Columns OK ({df.shape[1]}): {file_path.name}")
            except Exception as e:
                self.logger.error(f"✗ Read error {file_path.name}: {e}")

    def validate_missing_columns(self):
        """Check for entirely missing columns"""
        good_dir = Path("Training_Raw_files_validated/Good_Raw")
        good_files = list(good_dir.glob("*.csv"))

        for file_path in good_files:
            try:
                df = pd.read_csv(file_path)

                # Check for completely empty columns
                empty_cols = df.columns[df.isna().all()].tolist()
                if empty_cols:
                    bad_dest = Path("Training_Raw_files_validated/Bad_Raw") / file_path.name
                    shutil.move(str(file_path), str(bad_dest))
                    self.validation_results["bad_files"].append(file_path.name)
                    self.validation_results["good_files"].remove(file_path.name)
                    self.logger.warning(f"✗ Empty columns {empty_cols}: {file_path.name}")
                    continue

                # Fix unnamed index column (common pandas issue)
                if 'Unnamed: 0' in df.columns:
                    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
                    df.to_csv(file_path, index=False)
                    self.logger.info(f"✓ Fixed Unnamed:0 -> index: {file_path.name}")

            except Exception as e:
                self.logger.error(f"✗ Validation error {file_path.name}: {e}")

    def archive_bad_files(self):
        """Move bad files to timestamped archive"""
        bad_dir = Path("Training_Raw_files_validated/Bad_Raw")
        if not bad_dir.exists() or not any(bad_dir.iterdir()):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(f"TrainingArchiveBadData/BadData_{timestamp}")
        archive_dir.mkdir(parents=True, exist_ok=True)

        for file_path in bad_dir.glob("*.csv"):
            dest = archive_dir / file_path.name
            shutil.move(file_path, dest)

        self.logger.info(f"Archived {len(list(archive_dir.glob('*.csv')))} bad files")

    def validate_all(self):
        """Run complete validation pipeline"""
        try:
            # Step 1: Load schema
            date_len, time_len, col_names, num_cols = self.values_from_schema()

            # Step 2: Validate filenames
            self.validate_filenames(date_len, time_len)

            # Step 3: Validate column counts
            self.validate_column_count(num_cols)

            # Step 4: Validate missing data
            self.validate_missing_columns()

            # Step 5: Archive bad files
            self.archive_bad_files()

            # Summary
            summary = {
                "total_files": len(self.validation_results["good_files"]) + len(self.validation_results["bad_files"]),
                "good_files": self.validation_results["good_files"],
                "bad_files": self.validation_results["bad_files"]
            }
            self.logger.info(f"VALIDATION COMPLETE: {summary}")
            return summary

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


# Usage
if __name__ == "__main__":
    validator = Raw_Data_validation("Training_Batch_Files")
    results = validator.validate_all()
    print(f"Good: {len(results['good_files'])}, Bad: {len(results['bad_files'])}")
