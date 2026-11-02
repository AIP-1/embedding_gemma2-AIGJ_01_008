#!/bin/bash
python3 ~/eg2-lab/measure.py
free -h | grep Mem
docker stats --no-stream --format '{{.Name}}  {{.MemUsage}}' | grep simfeed
