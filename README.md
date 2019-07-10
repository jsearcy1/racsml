# Getting Started with Talapas
Coming Soon


# Getting Started with Docker
1) Make sure you have Docker installed https://www.docker.com/get-started
2) Clone repository in a terminal
`git clone https://github.com/jsearcy1/racsml.git`
3) switch to the created directory
 `cd racsml`
 4) Build the docker image
 `docker build $PWD -t racsml`
 5) At this point you should be able to see a docker image named racsml
```
docker images

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
racsml              latest              885ada57256a        27 minutes ago      1.79GB
python              3.7                 4c0fd7901be8        46 hours ago        929MB
```
6) Run the image
```
docker run -it -p 8888:8888 racsml
    
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-7-open.html
    Or copy and paste one of these URLs:
        http://(2e818c6cf771 or 127.0.0.1):8888/?token=d8ceec90df55c0a6cd2ab41669c3e8af5698f74f7904e2f4
```
        
7) Point your browser to http://127.0.0.1:8888/ and you'll be asked to login with the token listed in the output of the run command just
after the ?token= in the case above it's d8ceec90df55c0a6cd2ab41669c3e8af5698f74f7904e2f4
