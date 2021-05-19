

build:
	dts build_utils aido-container-build --use-branch daffy --ignore-untagged --force-login


push: build
	dts build_utils aido-container-push --use-branch daffy


submit: 
	dts challenges submit

submit-bea:
	dts challenges submit --impersonate 1639 --challenge all --retire-same-label
