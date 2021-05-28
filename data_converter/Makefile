# Make file for easier to start the conversion container
image:
				docker build -t extract_container .; \

extract:
	docker run -it --rm \
	-v `pwd`/bag_files:/workspace/bag_files/ \
	-v `pwd`/converted:/workspace/converted/ \
	extract_container ; \

start_record:
	dts duckiebot demo --demo_name make_log_docker --duckiebot_name ${DUCKIEBOT_NAME} --package_name duckietown_demos

stop_record:
	docker -H ${DUCKIEBOT_NAME}.local stop demo_make_log_docker

start_lane:
	dts duckiebot demo --demo_name lane_following --duckiebot_name ${DUCKIEBOT_NAME} --package_name duckietown_demos
	dts duckiebot keyboard_control ${DUCKIEBOT_NAME}

stop_lane:
	docker -H ${DUCKIEBOT_NAME}.local stop demo_lane_following
