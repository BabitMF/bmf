#!/bin/sh

JARDIR=".:/root/.m2/repository/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar:../../java/target/java-1.0-SNAPSHOT.jar:../../../bmf/hml/java/target/java-1.0-SNAPSHOT.jar"

echo "compile java"
javac -cp $JARDIR $1.java

echo "run java"
java -cp $JARDIR $1