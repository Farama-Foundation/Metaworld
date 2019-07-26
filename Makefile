SHELL := /bin/bash

.PHONY: help test ci-job build-ci build-headlessrun-ci run-headless

.DEFAULT_GOAL := help

# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ~/.mujoco/mjkey.txt

test:  ## Run the CI test suite locally
test: RUN_CMD = pytest -v
test: run
	@echo "Running test suite..."

ci-job:
	pytest -n $$(nproc) --cov=metaworld -v
	coverage xml
	bash <(curl -s https://codecov.io/bash)

ci-deploy-docker:
	echo "${DOCKER_API_KEY}" | docker login -u "${DOCKER_USERNAME}" \
		--password-stdin
	docker tag "${TAG}" ryanjulian/metaworld-ci:latest
	docker push ryanjulian/metaworld-ci

build-ci: TAG ?= ryanjulian/metaworld-ci:latest
build-ci: docker/docker-compose.yml
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose.yml \
		build \
		${ADD_ARGS}

run-ci: TAG ?= ryanjulian/metaworld-ci
run-ci:
	docker run \
		-e TRAVIS_BRANCH \
		-e TRAVIS_PULL_REQUEST \
		-e TRAVIS_COMMIT_RANGE \
		-e TRAVIS \
		-e MJKEY \
		${ADD_ARGS} \
		${TAG} ${RUN_CMD}

run: ## Run the Docker container on a local machine
run: CONTAINER_NAME ?= metaworld-ci
run: build-ci
	docker run \
		-it \
		--rm \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${ADD_ARGS} \
		ryanjulian/metaworld-ci $(RUN_CMD)

# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
