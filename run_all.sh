#!/bin/bash
for algo in FedAvg FedProx FedPer FedMAML FedGA FedGA_Meta; do
  for local_epochs in 1 5 10; do
    echo "Running $algo with local_epochs=$local_epochs"
    python Algorithms/${algo}.py --local_epochs $local_epochs --epochs 52 || exit 1
  done
done
