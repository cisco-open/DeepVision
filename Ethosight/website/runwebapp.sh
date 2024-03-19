
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
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

python3 "$SCRIPT_DIR/manage.py" makemigrations
python3 "$SCRIPT_DIR/manage.py" migrate
python3 "$SCRIPT_DIR/manage.py" createsuperuser
python3 "$SCRIPT_DIR/manage.py" runserver 8080
