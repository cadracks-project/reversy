#!/usr/bin/env bash

xhost +local:reversy
docker start reversy
docker exec -it reversy /bin/bash
