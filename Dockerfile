# Use an official python runtime as a parent image
FROM python:3.10-slim

# set working directory to /app
WORKDIR /app

# copy contents of 'testingphase' into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip


# install python dependencies
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME world

# Run app.py when the container launches
CMD ["python", "flask_react_app/app.py"]
