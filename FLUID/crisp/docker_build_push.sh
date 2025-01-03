#!/usr/bin/env bash
if docker build -t ah-crisp-validate .  ; then
  docker tag ah-crisp-validate gcr.io/fdl-us-astronaut-health/ah-crisp-validate
  docker push gcr.io/fdl-us-astronaut-health/ah-crisp-validate

#  docker tag ah-causal-ensemble-docker registry.gitlab.com/frontierdevelopmentlab/astronaut-health/ah-causal-ensemble
#  docker push registry.gitlab.com/frontierdevelopmentlab/astronaut-health/ah-causal-ensemble
fi
