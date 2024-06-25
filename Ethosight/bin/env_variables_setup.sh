
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

# EthosightBackend type client
# Not modifiable
export EthosightBackend=client
# EthosightBackendURL is pointing to nginx running
export EthosightBackendURL=http://localhost:80
# DjangoEthosightAppBaseDir points to the app created on ui side
export DjangoEthosightAppBaseDir=/home/vahagn/projects/EthosightNew/website/EthosightAppBasedir
# EthosightYAMLDirectory points the created config yaml file location
export EthosightYAMLDirectory=/home/vahagn/projects/EthosightNew/configs
# ETHOSIGHT_APP_BASEDIR points to ethosight app created
export ETHOSIGHT_APP_BASEDIR=/home/vahagn/projects/EthosightNew/website/EthosightAppBasedir
# Email sending variables to send access codes to client users. You can modify as per your environment
export EMAIL_HOST=email-smtp.us-east-1.amazonaws.com
export EMAIL_PORT=587
export EMAIL_USE_TLS=True
export EMAIL_USE_SSL=False
export EMAIL_HOST_USER=''
export EMAIL_HOST_PASSWORD=''