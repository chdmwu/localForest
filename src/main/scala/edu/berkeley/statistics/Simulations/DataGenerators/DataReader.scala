package edu.berkeley.statistics.Simulations.DataGenerators

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Created by christopher on 2/8/16.
 */
object DataReader {
  def getData(sc: SparkContext, simulationName: String, nPartitions: Int, nTrain:Int, nValid: Int, nTest: Int, nActive: Int = 0, nInactive: Int = 0, nBasis:Int = 2500, normalizeLabel:Boolean = true, normalizeFeatures:Boolean = true, seed: Int = 333):
  (RDD[LabeledPoint], IndexedSeq[LabeledPoint],IndexedSeq[LabeledPoint]) =  {
    var d = -1;
    var (trainingDataRDD, validData, testData) = simulationName match {
      case "Friedman1" => {
        d = 5 + nInactive
        val dataGenerator = Friedman1Generator(nInactive)
        val seeds = Array.fill(nPartitions)(scala.util.Random.nextLong())
        val seedsRDD = sc.parallelize(seeds, nPartitions)
        val trainingDataRDD = seedsRDD.flatMap(
          seed => dataGenerator.generateData(
            nTrain/nPartitions, 3.0, new scala.util.Random(seed)))
        val validData = dataGenerator.
          generateData(nValid, 0.0, scala.util.Random)
        val testData = dataGenerator.
          generateData(nTest, 0.0, scala.util.Random)
        (trainingDataRDD, validData, testData)
      }
      case "GaussianProcess" => {
        d = nActive + nInactive
        val function = FourierBasisGaussianProcessFunction.getFunction(
          nActive, nBasis, 0.05, new scala.util.Random(2015L))
        val dataGenerator = new FourierBasisGaussianProcessGenerator(function, nInactive)
        val seeds = Array.fill(nPartitions)(scala.util.Random.nextLong())
        val seedsRDD = sc.parallelize(seeds, nPartitions)
        val trainingDataRDD = seedsRDD.flatMap(
          seed => dataGenerator.generateData(
            nTrain/nPartitions, 3.0, new scala.util.Random(seed)))
        val validData = dataGenerator.
          generateData(nValid, 0.0, scala.util.Random)
        val testData = dataGenerator.
          generateData(nTest, 0.0, scala.util.Random)
        (trainingDataRDD, validData, testData)
      }
      case x => {
        val dataRDD = sc.textFile(x, nPartitions*3).map(line => createLabeledPoint(line))//a hack to read into nPartitions
        val totalPoints = nTrain + nValid + nTest
        val trainFrac = nTrain.toDouble/totalPoints
        val validFrac = nValid.toDouble/totalPoints
        val testFrac = nTest.toDouble/totalPoints

        val splitData = dataRDD.randomSplit(Array(trainFrac, validFrac, testFrac), seed)

        //trainingDataRDD.
        (splitData(0).coalesce(nPartitions), splitData(1).collect().toIndexedSeq, splitData(2).collect().toIndexedSeq)//a hack to read into nPartitions
      }
    }
   // val nTrain = trainingDataRDD.count()
    val trMean = trainingDataRDD.map(point => point.label).sum/nTrain
    val trVariance = trainingDataRDD.map(point => Math.pow(point.label - trMean, 2)).sum / nTrain
    val trStdev = Math.sqrt(trVariance)
    if(normalizeLabel){
      trainingDataRDD = trainingDataRDD.map(point => new LabeledPoint((point.label-trMean)/trStdev, point.features))
      validData = validData.map(point => new LabeledPoint((point.label-trMean)/trStdev, point.features))
      testData = testData.map(point => new LabeledPoint((point.label-trMean)/trStdev, point.features))
    }
    if(normalizeFeatures){
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainingDataRDD.map(x => x.features))
      trainingDataRDD = trainingDataRDD.map(point => new LabeledPoint(point.label, scaler.transform(point.features)))
      validData = validData.map(point => new LabeledPoint(point.label, scaler.transform(point.features)))
      testData = testData.map(point => new LabeledPoint(point.label, scaler.transform(point.features)))
    }
    (trainingDataRDD, validData, testData)
  }
  def createLabeledPoint(line: String) : LabeledPoint = {
    val tokens = line.split(",").map(_.toDouble)
    return new LabeledPoint(tokens.last, Vectors.dense(tokens.dropRight(1)))
  }
}
