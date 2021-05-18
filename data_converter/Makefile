# Make file for easier to start the conversion container
image:
				docker build -t extract_container .; \

extract:
	docker run -it --rm \
	-v `pwd`/bag_files:/workspace/bag_files/ \
	-v `pwd`/converted:/workspace/converted/ \
	extract_container ; \
