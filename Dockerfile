FROM guillaumeflorent/miniconda-pythonocc:3-0.18.3

MAINTAINER Guillaume Florent <florentsailing@gmail.com>

#############
# pythreejs #
#############

RUN apt-get update && apt-get install -y npm && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/jovyan/pythreejs
WORKDIR /opt/pythreejs
RUN /opt/conda/bin/pip install .
WORKDIR /opt/pythreejs/js
RUN npm run autogen
RUN npm run build:all
RUN jupyter nbextension install --py --symlink --sys-prefix pythreejs
RUN jupyter nbextension enable pythreejs --py --sys-prefix

# For wx : libgtk2.0-0 libxxf86vm1
# Funily, installing libgtk2.0-0 seems to solve the XCB plugin not found issue for Qt !!
# For pyqt : libgl1-mesa-dev libx11-xcb1
RUN apt-get update && apt-get install -y libgtk2.0-0 libxxf86vm1 libgl1-mesa-dev libx11-xcb1 && rm -rf /var/lib/apt/lists/*

# Other conda packages
RUN conda install -y numpy matplotlib networkx pandas wxpython pyqt pytest

# ccad
WORKDIR /opt
# ADD https://api.github.com/repos/osv-team/ccad/git/refs/heads/master version.json
RUN git clone --depth=1 https://github.com/osv-team/ccad
WORKDIR /opt/ccad
RUN python setup.py install

# reversy
WORKDIR /opt
# ADD https://api.github.com/repos/osv-team/reversy/git/refs/heads/master version.json
RUN git clone --depth=1 https://github.com/osv-team/reversy
WORKDIR /opt/reversy
RUN python setup.py install