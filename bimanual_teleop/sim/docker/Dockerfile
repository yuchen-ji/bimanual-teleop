FROM nvcr.io/nvidia/isaac-sim:5.1.0
COPY requirements.txt /tmp/sim-requirements.txt
COPY install_packages.py /tmp/sim-install-packages.py
RUN /isaac-sim/python.sh /tmp/sim-install-packages.py install --no-deps -r /tmp/sim-requirements.txt
ENV PYTHONPATH=/workspace/bimanual-teleop
WORKDIR /isaac-sim
ENTRYPOINT ["/isaac-sim/python.sh", "/workspace/bimanual-teleop/bimanual_teleop/sim/run.py", "engine"]
