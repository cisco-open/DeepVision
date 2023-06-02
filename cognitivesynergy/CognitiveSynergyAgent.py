import redis
import json
import threading
import time
from ONA.DockerInteractor import DockerInteractor
from utils.DVDisplayChannel import DVDisplayChannel
import random

class CognitiveSynergyAgent:
    def __init__(self, agent_type, config):
        self.agent_type = agent_type
        self.config = config
        self.is_running = False
        self._thread = None
        self.ona = DockerInteractor(['docker', 'exec', '-i', 'ONA', '/app/NAR', 'shell'])
        self.display_channel = DVDisplayChannel("ONA_DISPLAY")

        # Set up Redis client
        #self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def _run(self):
        iterations = 0
        print(f"Agent of type {self.agent_type} is starting...")

        while self.is_running:
            # Process data and publish results to Redis
            data = {}  # Your data here
            #self.redis_client.publish('channel_name', json.dumps(data))
            iterations = iterations + 1
            #print(f"Agent of type {self.agent_type} is running...iteration {iterations}")
            x = 10
            y = 40
            self.display_channel.write_message(f"Agent of type {self.agent_type} is running...iteration {iterations}", text_position={'x': x, 'y': y}, bounding_box={'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}, color='red', line_width=2, font_size=20, font_color='white')
            response = self.ona.execute_command("<cat --> furry_animal>.\n<cat --> furry_animal>?\n0\n", "done with 0 additional inference steps")
            #print(f"Response: {response}")
            time.sleep(.1)

        print(f"Agent of type {self.agent_type} has stopped.")

    def start(self):
        # Start the agent
        self.is_running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        # Stop the agent
        self.is_running = False
        if self._thread is not None:
            self._thread.join()
