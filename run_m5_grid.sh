#!/bin/bash

# copy playbooks to ansible folder
sudo cp run_m5_grid_playbook.yml /etc/ansible/run_m5_grid_playbook.yml

# run ansible run file
ansible-playbook /etc/ansible/run_m5_grid_playbook.yml
