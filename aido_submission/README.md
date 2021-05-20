# aido_submission

This directory contains submission files and a final trained model to be used by the agent being evaluated. Its structure must match the expected submission format which is documented in more detail here: https://docs.duckietown.org/daffy/AIDO/out/index.html

## Usage

First register for a Duckietown account and [DockerHub](https://hub.docker.com/) account.
```
dts challenges config --docker-username <USERNAME> --docker-password <PASSWORD>
docker login
```

Fetch any models that aren't included in the repo to keep the repo size small:
```
./fetch_models.sh
```

Submit to remote leaderboard for evaluation:
```
dts challenges submit
```

## Local testing

Test locally, note that only some challenges allow local testing, so you might need to change `submission.yaml`:
```
dts challenges evaluate
```

## Real robot execution

Run on real robot (only works on Ubuntu):
```
dts duckiebot evaluate --duckiebot_name DUCKIEBOT_NAME
```

## AIDO results

1. baseline:
    - https://challenges.duckietown.org/v4/humans/submissions/15504
1. v0
    - https://challenges.duckietown.org/v4/humans/submissions/15509
    - https://challenges.duckietown.org/v4/humans/submissions/15512
