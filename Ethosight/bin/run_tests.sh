
# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
