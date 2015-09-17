name := "DistributedForest"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.5.0" % "provided"
)

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test"

mainClass in (Compile, run) := Some(
  "edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation")