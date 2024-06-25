
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

# Check if the right number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./copy_files.sh <number_of_files> <starting_line_number>"
    exit 1
fi

num_files=$1
start_line=$2

rm ../Abuse001_x264/*

# Copy the specified number of files starting from the given line number
for file in $(ls | sort -t "_" -k 2 -n | sed -n "$start_line,$(($start_line + $num_files - 1))p"); do 
    cp $file ../Abuse001_x264 ; 
done

