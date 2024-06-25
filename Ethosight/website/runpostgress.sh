
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

# Set the desired database name
DB_NAME="mydatabase"

if [ $(docker ps -a -f name=djangopostgres | grep -w djangopostgres | wc -l) -eq 1 ]; then
    docker start djangopostgres
else
    docker run --name djangopostgres \
           -e POSTGRES_PASSWORD=mysecretpassword \
           -e POSTGRES_DB=$DB_NAME \
           -p 5432:5432 \
           -d postgres
fi
