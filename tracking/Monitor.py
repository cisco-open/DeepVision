import subprocess
import time

class GPUCalculator:
    def __init__(self, redis_conn):
        self.count = 0
        self.redis_conn = redis_conn

    def add(self):
        self.count=self.count+1
        if(self.count%15==0):
            cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode('utf-8')
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
