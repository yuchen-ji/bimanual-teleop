FROM nvcr.io/nvidia/isaac-sim:5.1.0
COPY requirements.txt /tmp/sim-requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-deps -r /tmp/sim-requirements.txt
ENV PYTHONPATH=/workspace/bimanual-teleop
WORKDIR /isaac-sim
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "bimanual_teleop.sim.engine"]
