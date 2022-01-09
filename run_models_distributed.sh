#!/bin/bash

# copy playbooks to ansible folder
sudo cp setup_playbook.yml /etc/ansible/setup_playbook.yml
sudo cp run_models_playbook.yml /etc/ansible/run_models_playbook.yml

# run ansible setup file
ansible-playbook /etc/ansible/setup_playbook.yml

# copy data files to servers
# this task needs to be preceded by the ansible setup
# because that is the moment that the directories are
# pulled from github fresh
scp -r data/ mach2ne:/home/mach2ne/run_models_distributed/
scp -r data/ coral:/home/lroque/run_models_distributed/
scp -r data/ marineye:/home/lroque/run_models_distributed/
scp -r data/ nitro:/home/lroque/run_models_distributed/

# run ansible run file
ansible-playbook /etc/ansible/run_models_playbook.yml
