#!/bin/bash
for i in {1..5}
do
  python runner.py configs/comparison100.ini
  python runner.py configs/partial100.ini
done
