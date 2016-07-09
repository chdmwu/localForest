package edu.berkeley.statistics.Simulations.DataGenerators

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Created by christopher on 2/8/16.
 */

//TODO GET RID OF NTRAIN NVALID NTEST ASAP


object DataReader {
  def getData(sc: SparkContext, trainFile: String, nPartitions: Int, nTrain:Int, nValid: Int, nTest: Int, validFile:String = "", testFile:String = "", nActive: Int = 0, nInactive: Int = 0, nBasis:Int = 2500, normalizeLabel:Boolean , normalizeFeatures:Boolean, seed: Int = 333):
  (RDD[LabeledPoint], IndexedSeq[LabeledPoint],IndexedSeq[LabeledPoint]) =  {
    var d = -1;
    val noisesd = 3.0 //TODO fix hardcoding of noise
    //var (trainingDataRDD, validData, testData) = trainFile match {
    var (trainingDataRDD, validData, testData) = trainFile match {
      case "Friedman1" => {
        d = 5 + nInactive
        val dataGenerator = Friedman1Generator(nInactive)
        val seeds = Array.fill(nPartitions)(scala.util.Random.nextLong())
        val seedsRDD = sc.parallelize(seeds, nPartitions)
        val trainingDataRDD = seedsRDD.flatMap(
          seed => dataGenerator.generateData(
            nTrain/nPartitions, noisesd, new scala.util.Random(seed)))
        val validData = dataGenerator.
          generateData(nValid, noisesd, scala.util.Random)
        val testData = dataGenerator.
          generateData(nTest, 0.0, scala.util.Random)
        (trainingDataRDD, validData, testData)
      }
      case "Linear" => {
        d = nActive + nInactive
        val beta = Array.fill(nActive)(20 * (scala.util.Random.nextDouble() - .5))
        println("beta: " + beta)
        val dataGenerator = LinearGenerator(nActive, nInactive, beta)
        val seeds = Array.fill(nPartitions)(scala.util.Random.nextLong())
        val seedsRDD = sc.parallelize(seeds, nPartitions)
        val trainingDataRDD = seedsRDD.flatMap(
          seed => dataGenerator.generateData(
            nTrain/nPartitions, noisesd, new scala.util.Random(seed)))
        val validData = dataGenerator.
          generateData(nValid, noisesd, scala.util.Random)
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
            nTrain/nPartitions, noisesd, new scala.util.Random(seed)))
        val validData = dataGenerator.
          generateData(nValid, noisesd, scala.util.Random)
        val testData = dataGenerator.
          generateData(nTest, 0.0, scala.util.Random)
        (trainingDataRDD, validData, testData)
      }
      case x => (testFile.length,validFile.length) match{
        //if no validation and test file are given, we split the training file
        case (0,0) => {
          val dataRDD = sc.textFile(x, nPartitions*3).map(line => createLabeledPoint(line))//a hack to read into nPartitions
          val totalPoints = nTrain + nValid + nTest
          val trainFrac = nTrain.toDouble/totalPoints
          val validFrac = nValid.toDouble/totalPoints
          val testFrac = nTest.toDouble/totalPoints

          val splitData = dataRDD.randomSplit(Array(trainFrac, validFrac, testFrac), seed)

          //trainingDataRDD.
          (splitData(0).coalesce(nPartitions), splitData(1).collect().toIndexedSeq, splitData(2).collect().toIndexedSeq)//a hack to read into nPartitions
        }
          // if we are given valid and test files
        case y => {
          val train = sc.textFile(trainFile, nPartitions).map(line => createLabeledPoint(line))
          val valid = sc.textFile(validFile).map(line => createLabeledPoint(line)).collect().toIndexedSeq
          val test = sc.textFile(testFile).map(line => createLabeledPoint(line)).collect().toIndexedSeq
          (train.coalesce(nPartitions), valid, test)
        }
      }
    }
   // val nTrain = trainingDataRDD.count()

    val true_nTrain = trainingDataRDD.count
    val trMean = trainingDataRDD.map(point => point.label).sum/ true_nTrain
    val trVariance = trainingDataRDD.map(point => Math.pow(point.label - trMean, 2)).sum / true_nTrain
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
