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