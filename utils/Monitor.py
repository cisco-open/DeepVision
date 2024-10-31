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

import subprocess
import time

from utils.constants import GPU_METRICS_CMD, MEMORY_USED, MEMORY_TOTAL, GPU_UTILIZATION, MEMORY_UTILIZATION, GPU_TEMP

class TSManager:
    def __init__(self, redis_conn):
        self.redis_conn = redis_conn

    def ts_add(self, key, value):
        self.redis_conn.execute_command('ts.add {} * {}'.format(key, value))


class GPUCalculator:
    def __init__(self, ts_manager, threshold):
        self.count = 0
        self.ts_manager = ts_manager
        self.threshold = threshold
    
    def get_first_line_or_original(self,input_string):
        # Split the string by newline characters
        lines = input_string.split('\n')
    
        # Check if there are more than one lines
        if len(lines) > 1:
            return lines[0]  # return the first line if there are multiple lines
    
        return input_string  # return the original string if there's only one line


    def add(self):
        self.count = self.count + 1
        if(self.count % self.threshold == 0):
            output = subprocess.check_output(GPU_METRICS_CMD, shell=True)
            output = output.decode('utf-8')
            output = self.get_first_line_or_original(output)
            gpu_stats = output.strip().split(', ')  
            memory_used = int(gpu_stats[0])
            memory_total = int(gpu_stats[1])
            gpu_utilization = int(gpu_stats[2])
            memory_utilization = int(gpu_stats[3])
            gpu_temp = int(gpu_stats[4])

            self.ts_manager.ts_add(MEMORY_USED, memory_used)
            self.ts_manager.ts_add(MEMORY_TOTAL, memory_total)
            self.ts_manager.ts_add(GPU_UTILIZATION, gpu_utilization)
            self.ts_manager.ts_add(MEMORY_UTILIZATION, memory_utilization)
            self.ts_manager.ts_add(GPU_TEMP, gpu_temp)
            

class MMTMonitor:
    def __init__(self, ts_manager, ts_key, threshold):
        self.start_time = None
        self.ts_manager = ts_manager
        self.ts_key = ts_key
        self.counter = 0
        self.average_latency = 0
        self.threshold = threshold
    def start_timer(self):
        self.start_time = time.time()
    def end_timer(self):
        self.counter = self.counter + 1
        end_time = time.time()
        latency = end_time - self.start_time
        if self.counter % self.threshold == 0:
            self.average_latency = self.average_latency/self.threshold
            self.ts_manager.ts_add(self.ts_key, self.average_latency)
            self.average_latency = 0
        else:
            self.average_latency = self.average_latency + latency


