# Ansible install
```commandline
python3 -m pip install ansible
```

# Run needed script

```bash
ansible-playbook -i inventory_local.ini <<playbook name>>
```
> **Note:**  
> Depending on your current user privileges you have to run some playbooks with root user (sudo) 
> If in a playbook *.yaml file it is mentioned 'become:yes', this means that it wants root user to execute.   

## Playbooks that you have to run depending on your state 

```commandline
ansible-playbook -i inventory_local.ini updateandupgrade.yaml
ansible-playbook -i inventory_local.ini install_ethosight.yaml
ansible-playbook -i inventory_local.ini install_docker.yaml
ansible-playbook -i inventory_local.ini install_conda.yaml
ansible-playbook -i inventory_local.ini create_conda_env.yaml
ansible-playbook -i inventory_local.ini install_consul.yaml
ansible-playbook -i inventory_local.ini install_nginx.yaml
ansible-playbook -i inventory_local.ini install_consul_template.yml
ansible-playbook -i inventory_local.ini install_and_run_gpuserver.yaml
ansible-playbook -i inventory_local.ini install_and_run_webserver.yaml
```

# Todo

- [ ] Move all variables into uservars.yml
- [ ] Write master script to run all needed sub playbooks
- [ ] Move all playbooks into subdirectory install/playbooks (master script stays in /install)
- [ ] Create a master script for installing gpu server nodes



