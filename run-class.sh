#!/usr/bin/env bash

JAR_FILE=target/scala-2.12/mllib_2.12-1.0.jar
sbt package && \
spark-submit --class $1 $JAR_FILE