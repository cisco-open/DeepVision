
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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

docker run --name tmp-nginx -d nginx
docker cp tmp-nginx:/etc/nginx /tmp
docker stop tmp-nginx
docker rm tmp-nginx

cp "$SCRIPT_DIR/../config/nginx/nginx.conf" /tmp/nginx/nginx.conf

while true ; do
    docker rm -f ethosight-nginx

    docker run --name ethosight-nginx --network host \
        -v /tmp/nginx:/etc/nginx:ro \
        -p 80:80 -t nginx

    sleep 5 
done

