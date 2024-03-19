
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

#!/usr/bin/env python3
import click
import os
import uvicorn
from time import sleep
from Ethosight.EthosightRESTServer import EthosightRESTServer  # Adjust the import path based on your directory structure

@click.group()
def cli():
    pass

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host for the server. Default is 127.0.0.1.')
@click.option('--port', default=8000, type=int, help='Port for the server. Default is 8000.')
@click.option('--log-level', default='debug', help='Logging level for the server. Default is "debug".')
@click.option('--mode', default='blocking', help='Mode for the server. Can be "blocking" or "non-blocking". Default is "blocking".')
@click.option('--consul-url', default='localhost', help='URL for Consul server. Default is "localhost".')
@click.option('--consul-port', default=8500, type=int, help='Port for Consul server. Default is 8500.')
@click.option('--gpu', default=0, type=int, help='GPU to use. Default is 0.')
@click.option('--reasoner', default='', help='Reasoner type like ChatGPTReasoner. Default is no reasoner')
def runserver(host, port, log_level, mode, consul_url, consul_port, gpu, reasoner):
    """
    Run the Ethosight REST Server with the specified options.
    """
    # Assuming the EthosightRESTServer class accepts the consul_url and consul_port as arguments.
    # If not, you'll need to adjust the instantiation accordingly.
    server = EthosightRESTServer(mode=mode, host=host, port=port, consul_url=consul_url, consul_port=consul_port, gpu=gpu, reasoner=reasoner)
    uvicorn.run(server.app, host=host, port=port, log_level=log_level)

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host for the server. Default is 127.0.0.1.')
@click.option('--port', default=8000, type=int, help='Port for the server. Default is 8000.')
@click.option('--log-level', default='debug', help='Logging level for the server. Default is "debug".')
@click.option('--mode', default='blocking', help='Mode for the server. Can be "blocking" or "non-blocking". Default is "blocking".')
@click.option('--consul-url', default='localhost', help='URL for Consul server. Default is "localhost".')
@click.option('--consul-port', default=8500, type=int, help='Port for Consul server. Default is 8500.')
@click.option('--gpu', default=0, type=int, help='GPU to use. Default is 0.')
def start_server(host, port, log_level, mode, consul_url, consul_port, gpu):
    """Start the Ethosight REST Server in a tmux session."""
    print(f"Starting server on {host}:{port} in {mode} mode with log level {log_level}...")
    session_name = f"EthosightRESTServer{port}"
    cmd = f"tmux new-session -d -s {session_name} > tmux_output.txt 2>&1"
    print("executing command: ", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error starting tmux session {session_name}.")
        return
    ret = os.system(f"tmux send-keys -t {session_name} 'source ~/env_ethosight && ./runserver.py runserver --host={host} --port={port} --log-level={log_level} --mode={mode} --consul-url={consul_url} --consul-port={consul_port} --gpu={gpu}' C-m")
    if ret != 0:
        print(f"Error starting server on {host}:{port} in {mode} mode with log level {log_level}.")
        return

@cli.command()
@click.option('--port', default=8000, type=int, help='Port for the server whose session is to be stopped. Default is 8000.')
def stop_server(port):
    """Stop the Ethosight REST Server and terminate the tmux session."""
    session_name = f"EthosightRESTServer{port}"
    os.system(f"tmux send-keys -t {session_name} C-c")  # Send CTRL+C to the tmux session to terminate the server process
    sleep(10)
    os.system(f"tmux kill-session -t {session_name}")


@cli.command()
def start_consul():
    """Start the Consul server in a tmux session."""
    os.system("tmux new-session -d -s consul_session")
    os.system("tmux send-keys -t consul_session 'exec consul agent -dev -ui' C-m")

@cli.command()
def stop_consul():
    """Stop the Consul server and terminate the tmux session."""
    os.system("tmux send-keys -t consul_session C-c")  # Send CTRL+C to the tmux session to terminate the Consul process
    sleep(10)
    os.system("tmux kill-session -t consul_session")

if __name__ == "__main__":
    cli()
