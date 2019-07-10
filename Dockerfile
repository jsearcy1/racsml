FROM python:3.7
COPY . /racsml
RUN cd /racsml && ls && pip install -r requirements.txt
EXPOSE 8888
CMD jupyter notebook --ip 0.0.0.0 --allow-root