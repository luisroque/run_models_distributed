#!/bin/bash

scp -r data/ mach2ne:/home/mach2ne/run_models_distributed/
scp -r data/ coral:/home/lroque/run_models_distributed/
scp -r data/ marineye:/home/lroque/run_models_distributed/
scp -r data/ nitro:/home/lroque/run_models_distributed/

ansible-playbook /etc/ansible/sample.yml
