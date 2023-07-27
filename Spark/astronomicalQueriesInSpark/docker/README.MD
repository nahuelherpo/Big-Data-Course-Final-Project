# PySpark with Docker Compose Tutorial

![Docker Logo](docker_logo.png)

This tutorial will guide you through the steps to set up PySpark with Docker Compose, allowing you to run PySpark applications in a containerized environment.

## Table of Contents

- [PySpark with Docker Compose Tutorial](#pyspark-with-docker-compose-tutorial)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
    - [Step 1: Install Dependencies](#step-1-install-dependencies)
    - [Step 2: Start the PySpark Container](#step-2-start-the-pyspark-container)
  - [Additional Resources](#additional-resources)

## Introduction

Provide a brief introduction to PySpark and Docker Compose.
Explain the benefits of using Docker Compose for running PySpark applications.


All in all, Apache Spark is an open source, distributed computing framework used for large data workloads, batch processing, and machine learning. In particular, Spark is a fast and developer friendly data processing tool from the Apache Software Foundation designed to enhance data intensive application performance. In short, it provides high level APIs in Java, Scala, Python, etc. to make it easy to build applications across a wide spectrum of use cases.

In short, Apache Spark is a fast, versatile engine for processing data at scale. Spark also takes some of the programming burden for these tasks off the developer’s shoulders, thanks to a simple to use API that abstracts out a lot of the grunt work in distributed computing and large scale data processing.

Many banks, gaming companies, government officials, tech giants, and telecommunication companies use Apache Spark for it supports SQL and helps in streaming data, batch processing, and machine learning.


Docker Compose offers significant advantages for managing multi-container applications in a streamlined manner. By utilizing a simple YAML configuration file, Docker Compose allows developers to define and orchestrate complex application environments effortlessly. With Compose, services can be started and stopped together as a cohesive unit, providing consistency and ease of deployment. Moreover, its ability to manage service dependencies and network configurations enhances collaboration among teams, as the entire application stack can be shared and executed uniformly across different environments. Docker Compose's efficiency in defining and scaling services empowers developers to optimize resource usage and adapt to varying workloads seamlessly. Ultimately, Docker Compose simplifies the process of building, deploying, and scaling containerized applications, fostering greater efficiency and consistency in modern software development workflows.


## Prerequisites

List the prerequisites required for this tutorial, including links to install Docker and Docker Compose if needed.
Include any other dependencies that need to be installed on the host system.

## Step 1: Install Dependencies

### 1. Install Dependencies, Docker and Docker Compose

Before starting, it is always a good idea to update all system packages to the updated version. Update all of them by running the following command:

```bash
$ apt update -y
$ apt upgrade -y
```

After updating all system packages, run the following command to install other required dependencies:

```bash
$ apt install apt-transport-https ca-certificates curl software-properties-common -y
```

Once you are done, then proceed to install Docker and Docker Compose.


The latest version of Docker is not available in the Ubuntu default repository. So you will need to install it from the Docker’s official repository.

First, import the Docker GPG key using the following command:

```bash
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
```

Next, add the Docker repository using the following command:

```bash
$ add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
```

Once the repository is added, you can install Docker and Docker Compose using the following command:

```bash
$ apt install docker-ce docker-compose -y
```

After the successful installation, verify the Docker version using the following command:

```bash
$ docker --version
```

Now see the Docker version in the following output:

```bash
Docker version 20.10.21, build baeda1f
```

Next, start the Docker service and enable it to start at system reboot:

```bash
$ systemctl start docker
$ systemctl enable docker
```

At this point, both Docker and Docker Compose is installed in your system. You can now proceed to the next step.




## Step 2: Start the PySpark Container

### 1. Build Apache Spark image

Run the following command to build the Apache Spark image:

```bash
$ docker build -t cluster-apache-spark:3.0.2 .
```

You should see the following output:

```bash
---> Running in 643f406b9920
Removing intermediate container 643f406b9920
 ---> 47097ac88c1a
Step 9/12 : EXPOSE 8080 7077 6066
 ---> Running in 4987c51e99af
Removing intermediate container 4987c51e99af
 ---> 5164f30db28b
Step 10/12 : RUN mkdir -p $SPARK_LOG_DIR && touch $SPARK_MASTER_LOG && touch $SPARK_WORKER_LOG && ln -sf /dev/stdout $SPARK_MASTER_LOG && ln -sf /dev/stdout $SPARK_WORKER_LOG
 ---> Running in 636c94c3035d
Removing intermediate container 636c94c3035d
 ---> e1e057d85b0a
Step 11/12 : COPY start-spark.sh /
 ---> b04caed7b9d7
Step 12/12 : CMD ["/bin/bash", "https://net.cloudinfrastructureservices.co.uk/start-spark.sh"]
 ---> Running in 3677d0d8abdb
Removing intermediate container 3677d0d8abdb
 ---> e9f89d95c667
Successfully built e9f89d95c667
Successfully tagged cluster-apache-spark:3.0.2
```

### 2. Run Docker Compose

Run the following command to launch the Apache Spark container:

```bash
$ docker-compose up -d
```

You should see the following output:
```bash
Pulling spark-master (docker.io/bitnami/spark:3.3)...
3.3: Pulling from bitnami/spark
9dce2fae8330: Pull complete
2f16a53695ed: Pull complete
Digest: sha256:fb8ff4a361bbf6eb1d213f4eca862a33d6d2506b138f4ec2ba106e968cde2118
Status: Downloaded newer image for bitnami/spark:3.3
Pulling demo-database (postgres:11.7-alpine)...
11.7-alpine: Pulling from library/postgres
cbdbe7a5bc2a: Pull complete
b52a8a2ca21a: Pull complete
e36a19831e31: Pull complete
f1aa26821845: Pull complete
412d098142b4: Pull complete
75d5ef10726d: Pull complete
ae3b5a8bbf62: Pull complete
e2f290791a5c: Pull complete
187b81308ed8: Pull complete
Digest: sha256:77fcd2c7fceea2e3b77e7a06dfc231e70d45cad447e6022346b377aab441069f
Status: Downloaded newer image for postgres:11.7-alpine
Creating root_demo-database_1 ... done
Creating root_spark-master_1  ... done
Creating root_spark-worker-b_1 ... done
Creating root_spark-worker-a_1 ... done
```

Verify all the running container, so please run the following command:

```bash
CONTAINER ID   IMAGE                  COMMAND                  CREATED              STATUS              PORTS                                                                                  NAMES
a29b1cf3062f   bitnami/spark:3.3      "/opt/bitnami/script…"   About a minute ago   Up About a minute   0.0.0.0:7000->7000/tcp, :::7000->7000/tcp, 0.0.0.0:9091->8080/tcp, :::9091->8080/tcp   root_spark-worker-a_1
8d5e6efb9c44   bitnami/spark:3.3      "/opt/bitnami/script…"   About a minute ago   Up About a minute   0.0.0.0:7001->7000/tcp, :::7001->7000/tcp, 0.0.0.0:9092->8080/tcp, :::9092->8080/tcp   root_spark-worker-b_1
bd1bd28315ea   bitnami/spark:3.3      "/opt/bitnami/script…"   About a minute ago   Up About a minute   0.0.0.0:7077->7077/tcp, :::7077->7077/tcp, 0.0.0.0:9090->8080/tcp, :::9090->8080/tcp   root_spark-master_1
ebf78d5fed73   postgres:11.7-alpine   "docker-entrypoint.s…"   About a minute ago   Up About a minute   0.0.0.0:5432->5432/tcp, :::5432->5432/tcp                                              root_demo-database_1
```

And to verify the downloaded images, run the following command:

```bash
REPOSITORY             TAG                       IMAGE ID       CREATED         SIZE
cluster-apache-spark   3.0.2                     e9f89d95c667   3 minutes ago   1.16GB
bitnami/spark          3.3                       a5187599fe89   2 days ago      1.23GB
openjdk                11.0.11-jre-slim-buster   f1d5c8a9bc51   17 months ago   220MB
postgres               11.7-alpine               36ff18d21807   2 years ago     150MB
```

### 3. Access Apache Spark

At this point, Apache spark is installed and running. Now, open your web browser and access the Apache Spark master using the URL http://your-server-ip:9090. You should see the Apache Spark master on the following screen:

![Apache Spark](https://net.cloudinfrastructureservices.co.uk/wp-content/uploads/2022/11/apache-spark-master.png)

If you have any problem to access any of the node. You can check the container logs using the following command:

```bash
$ docker-compose logs
```

You should see the following output:

```bash
spark-worker-b_1  |  11:32:01.78 INFO  ==> ** Starting Spark in master mode **
spark-worker-b_1  | rsync from spark://spark-master:7077
spark-worker-b_1  | /opt/bitnami/spark/sbin/spark-daemon.sh: line 177: rsync: command not found
spark-worker-b_1  | starting org.apache.spark.deploy.master.Master, logging to /opt/bitnami/spark/logs/spark--org.apache.spark.deploy.master.Master-1-8d5e6efb9c44.out
spark-worker-b_1  | Spark Command: /opt/bitnami/java/bin/java -cp /opt/bitnami/spark/conf/:/opt/bitnami/spark/jars/* -Xmx1g org.apache.spark.deploy.master.Master --host 8d5e6efb9c44 --port 7077 --webui-port 8080
spark-worker-b_1  | ========================================
spark-worker-b_1  | Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
spark-worker-b_1  | 22/11/24 11:32:08 INFO Master: Started daemon with process name: 44@8d5e6efb9c44
spark-worker-b_1  | 22/11/24 11:32:08 INFO SignalUtils: Registering signal handler for TERM
spark-worker-b_1  | 22/11/24 11:32:08 INFO SignalUtils: Registering signal handler for HUP
spark-worker-b_1  | 22/11/24 11:32:08 INFO SignalUtils: Registering signal handler for INT
spark-worker-b_1  | 22/11/24 11:32:11 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
spark-worker-b_1  | 22/11/24 11:32:11 INFO SecurityManager: Changing view acls to: spark
spark-worker-b_1  | 22/11/24 11:32:11 INFO SecurityManager: Changing modify acls to: spark
spark-worker-b_1  | 22/11/24 11:32:11 INFO SecurityManager: Changing view acls groups to:
spark-worker-b_1  | 22/11/24 11:32:11 INFO SecurityManager: Changing modify acls groups to:
spark-worker-b_1  | 22/11/24 11:32:11 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users  with view permissions: Set(spark); groups with view permissions: Set(); users  with modify permissions: Set(spark); groups with modify permissions: Set()
spark-worker-b_1  | 22/11/24 11:32:13 INFO Utils: Successfully started service 'sparkMaster' on port 7077.
spark-worker-b_1  | 22/11/24 11:32:13 INFO Master: Starting Spark master at spark://8d5e6efb9c44:7077
spark-worker-b_1  | 22/11/24 11:32:13 INFO Master: Running Spark version 3.3.1
spark-worker-b_1  | 22/11/24 11:32:14 INFO Utils: Successfully started service 'MasterUI' on port 8080.
spark-worker-b_1  | 22/11/24 11:32:14 INFO MasterWebUI: Bound MasterWebUI to spark-worker-b, and started at http://8d5e6efb9c44:8080
spark-worker-b_1  | 22/11/24 11:32:15 INFO Master: I have been elected leader! New state: ALIVE
```



To stop the Apache Spark, run the following command:

```bash
$ docker-compose down
```

This will stop all containers as shown below:

```bash
$ Stopping root_spark-worker-a_1 ... done
Stopping root_spark-worker-b_1 ... done
Stopping root_spark-master_1   ... done
Stopping root_demo-database_1  ... done
Removing root_spark-worker-a_1 ... done
Removing root_spark-worker-b_1 ... done
Removing root_spark-master_1   ... done
Removing root_demo-database_1  ... done
Removing network root_default
```


## Additional Resources

Create Apache Spark Docker Container using Docker-Compose:

https://cloudinfrastructureservices.co.uk/create-apache-spark-docker-container-using-docker-compose/ 