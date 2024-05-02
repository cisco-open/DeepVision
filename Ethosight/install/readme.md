# Ansible install
Already included in updated environment.yml so update conda with the latest version

# Run needed script

```bash
ansible-playbook -i inventory_local.ini <<playbook name>>
```

# Todo

- [ ] Move all variables into uservars.yml
- [ ] Write master script to run all needed sub playbooks
- [ ] Move all playbooks into subdirectory install/playbooks (master script stays in /install)
- [ ] Create a master script for installing gpu server nodes



