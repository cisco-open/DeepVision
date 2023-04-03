FROM grafana/grafana:latest-ubuntu

# Disable Login form or not
ENV GF_AUTH_DISABLE_LOGIN_FORM "true"
# Allow anonymous authentication or not
ENV GF_AUTH_ANONYMOUS_ENABLED "true"
# Role of anonymous user
ENV GF_AUTH_ANONYMOUS_ORG_ROLE "Admin"
# Install plugins here our in your own config file
ENV GF_INSTALL_PLUGINS="redis-app,redis-explorer-app"


# Add provisioning
ADD --chmod=755 provisioning /etc/grafana/provisioning

# Add configuration file
ADD --chmod=755 grafana.ini /etc/grafana/grafana.ini

# Add dashboard json files
ADD --chmod=755 dashboards /etc/grafana/dashboards

