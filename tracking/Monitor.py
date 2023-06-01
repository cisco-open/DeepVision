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

class GPUCalculator:
    def __init__(self, redis_conn):
        self.count = 0
        self.redis_conn = redis_conn
    
    def get_first_line_or_original(self,input_string):
        # Split the string by newline characters
        lines = input_string.split('\n')
    
        # Check if there are more than one lines
        if len(lines) > 1:
            return lines[0]  # return the first line if there are multiple lines
    
        return input_string  # return the original string if there's only one line


    def add(self):
        self.count=self.count+1
        if(self.count%15==0):
            cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode('utf-8')
            output = self.get_first_line_or_original(output)
            gpu_stats = output.strip().split(', ')  
            memory_used = int(gpu_stats[0])
            memory_total = int(gpu_stats[1])
            gpu_utilization = int(gpu_stats[2])
            memory_utilization = int(gpu_stats[3])
            gpu_temp = int(gpu_stats[4])
            self.redis_conn.execute_command('ts.add memory_used * {}'.format(memory_used))
            self.redis_conn.execute_command('ts.add memory_total * {}'.format(memory_total))
            self.redis_conn.execute_command('ts.add gpu_utilization * {}'.format(gpu_utilization))
            self.redis_conn.execute_command('ts.add memory_utilization * {}'.format(memory_utilization))
            self.redis_conn.execute_command('ts.add gpu_temp * {}'.format(gpu_temp))
            

class MMTMonitor:
    def __init__(self, redis_conn, redis_key):
        self.start_time = None
        self.redis_conn = redis_conn
        self.redis_key = redis_key
        self.counter = 0 
        self.average_latency = 0 
    def start_timer(self):
        self.start_time = time.time()
    def end_timer(self):
        self.counter = self.counter + 1
        end_time = time.time()
        latency = end_time - self.start_time
        if (self.counter%15==0):
            self.average_latency = self.average_latency/15
            self.redis_conn.execute_command('ts.add {} * {}'.format(self.redis_key,self.average_latency))
            self.average_latency = 0
        else:
            self.average_latency = self.average_latency + latency
