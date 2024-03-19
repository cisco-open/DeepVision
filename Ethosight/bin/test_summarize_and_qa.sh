
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

./EthosightCLI.py summarize --label-affinity-scores shoplifting.affinities -o shoplifting_summary.txt
./EthosightCLI.py ask images/shoplifting.png --background-knowledge "the man is bill gates" --summary-file shoplifting_summary.txt --questions "what is his name?
what did he do?
which hair color does he have?
is he man or woman?
" --outfile shoplifting_answers.txt
