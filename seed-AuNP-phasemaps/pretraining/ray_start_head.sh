#!/bin/bash

echo "starting ray head node"
# Launch the head node
ray start --head --node-ip-address=$1 --port=6379 --redis-password=$2
sleep infinity