#!/bin/bash

# Create a new file for the test report
report_file="test_report.txt"
echo "Test Report" > $report_file
echo "-----------" >> $report_file

# Run Python tests
echo "Running Python tests..."
pytest --ignore=tests/archivedtests --junitxml=python_test_report.xml tests/

# Find and run shell script tests
echo "Running shell script tests..."
find tests/ -type d -name 'archivedtests' -prune -o -type f -name 'test_*.sh' -print0 | while IFS= read -r -d '' file; 
do
    # Save the current directory.
    pushd . > /dev/null

    # Change to the directory of the script.
    cd "$(dirname "$file")"

    echo "Running $file"
    bash "$(basename "$file")"
    if [ $? -eq 0 ]
    then
      echo "$file completed successfully"
      echo "$file: PASS" >> "../../$report_file"
    else
      echo "Error in $file" >&2
      echo "$file: FAIL" >> "../../$report_file"
    fi

    # Restore the original directory.
    popd > /dev/null
done
