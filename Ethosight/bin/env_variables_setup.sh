#!/bin/bash

# EthosightBackend value must be core or client
export EthosightBackend=core
# EthosightBackendURL is pointing to nginx running
export EthosightBackendURL=http://localhost:80
# DjangoEthosightAppBaseDir points to the app created on ui side
export DjangoEthosightAppBaseDir=/home/vahagn/projects/EthosightNew/website/EthosightAppBasedir
# EthosightYAMLDirectory points the created config yaml file location
export EthosightYAMLDirectory=/home/vahagn/projects/EthosightNew/configs
# ETHOSIGHT_APP_BASEDIR points to ethosight app created
export ETHOSIGHT_APP_BASEDIR=/home/vahagn/projects/EthosightNew/website/EthosightAppBasedir
