package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{FeatureImportance, RandomForestParameters, TreeParameters, RandomForest}
import edu.berkeley.statistics.Simulations.DataGenerators.{FourierBasisGaussianProcessFunction, FourierBasisGaussianProcessGenerator, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row}
import scala.math.ceil

object ExecuteDistributedSimulation {
  val usage =
    """ExecuteDistributedSimulation --simulationName [name] --nPartitions [val] --nPointsPerPartition [val] --nPnnsPerPartition [val]
      |--nTest [val] --batchSize [val] --nActive [val] --nInactive [val] --nBasis [val] --ntree [val --mtry [val] --minNodeSize [val] --runGlobalRF [val]
      |--sampleWithReplacement [val]
    """.stripMargin
  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("DistributedForest")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)


    // For large simulations, may have to edit /usr/local/etc/sbtopts on MacOS if using sbt
    // or add --executor-memory XXG commandline option if using spark-submit
    if (args.length == 0) println(usage)
    val arglist = args.toList
    type OptionMap = Map[Symbol, Any]

    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      list match {
        case Nil => map
        case "--simulationName" :: value :: tail =>
          nextOption(map ++ Map('simulationName -> value.toString), tail)
        case "--nPartitions" :: value :: tail =>
          nextOption(map ++ Map('nPartitions -> value.toInt), tail)
        case "--nPointsPerPartition" :: value :: tail =>
          nextOption(map ++ Map('nPointsPerPartition -> value.toInt), tail)
        case "--nPnnsPerPartition" :: value :: tail =>
          nextOption(map ++ Map('nPnnsPerPartition -> value.toInt), tail)
        case "--nTest" :: value :: tail =>
          nextOption(map ++ Map('nTest -> value.toInt), tail)
        case "--batchSize" :: value :: tail =>
          nextOption(map ++ Map('batchSize -> value.toInt), tail)
        case "--nActive" :: value :: tail =>
          nextOption(map ++ Map('nActive -> value.toInt), tail)
        case "--nInactive" :: value :: tail =>
          nextOption(map ++ Map('nInactive -> value.toInt), tail)
        case "--nBasis" :: value :: tail =>
          nextOption(map ++ Map('nBasis -> value.toInt), tail)
        case "--mtry" :: value :: tail =>
          nextOption(map ++ Map('mtry -> value.toInt), tail)
        case "--ntree" :: value :: tail =>
          nextOption(map ++ Map('ntree -> value.toInt), tail)
        case "--minNodeSize" :: value :: tail =>
          nextOption(map ++ Map('minNodeSize -> value.toInt), tail)
        case "--runGlobalRF" :: value :: tail =>
          nextOption(map ++ Map('runGlobalRF -> value.toInt), tail)
        case "--runOracle" :: value :: tail =>
          nextOption(map ++ Map('runOracle -> value.toInt), tail)
        case "--sampleWithReplacement" :: value :: tail =>
          nextOption(map ++ Map('sampleWithReplacement -> value.toInt), tail)
        case "--threshold1" :: value :: tail =>
          nextOption(map ++ Map('threshold1 -> value.toDouble), tail)
        case "--threshold2" :: value :: tail =>
          nextOption(map ++ Map('threshold2 -> value.toDouble), tail)
        case "--nValid" :: value :: tail =>
          nextOption(map ++ Map('nValid -> value.toInt), tail)

        case option :: tail => System.err.println("Unknown option "+option)
          System.err.println(usage)
          sys.exit(1)
      }
    }
    def getArg(args: Map[Symbol, Any], defaultArgs: Map[Symbol, Any], name: Symbol) : Any = args(name) match {
      case -1 => defaultArgs(name)
      case _ => args(name)
    }
    def castInt(value: Any) : Int = value match {
      case x: Int => x
      case x: Double => x.toInt
      case _ => System.err.println("invalid parameter: " + value)
        sys.exit(1)
    }
    def castDouble(value: Any) : Double = value match {
      case x: Double => x
      case x: Int => x.toDouble
      case _ => System.err.println("invalid parameter: " + value)
        sys.exit(1)
    }
    val defaultArgs = Map('simulationName -> "Friedman1", 'nPartitions ->4, 'nPointsPerPartition ->2500, 'nPnnsPerPartition ->1000,
      'nTest ->1000,'batchSize -> 100,'nActive ->5,'nInactive ->0, 'nBasis -> 2500, 'ntree -> 100, 'minNodeSize -> 10, 'sampleWithReplacement -> 1
      , 'runOracle -> -1, 'threshold1 -> .1, 'threshold2 -> .33, 'nValid -> 1000).withDefaultValue(-1);
    val options = nextOption(Map().withDefaultValue(-1),arglist)
    if(options('simulationName) == "Friedman1"){
      if(options('nActive) != -1 && options('nActive) != 5){
        System.err.println("Warning: Friedman1 has 5 active dimensions, user supplied nActive ignored.")
      }
    }
    if(options('simulationName) != "GaussianProcess"){
      if(options('nBasis) != -1){
        System.err.println("Warning: nBasis only applicable to GaussianProcess dataset, user supplied nBasis ignored.")
      }
    }

    val simulationName = getArg(options, defaultArgs, 'simulationName)
    val nPartitions = castInt(getArg(options, defaultArgs, 'nPartitions))
    val nPointsPerPartition = castInt(getArg(options, defaultArgs, 'nPointsPerPartition))
    val nPnnsPerPartition = castInt(getArg(options, defaultArgs, 'nPnnsPerPartition))
    val nTest = castInt(getArg(options, defaultArgs, 'nTest))
    val batchSize = castInt(getArg(options, defaultArgs, 'batchSize))
    val nActive = castInt(getArg(options, defaultArgs, 'nActive))
    val nInactive = castInt(getArg(options, defaultArgs, 'nInactive))
    val nBasis =  castInt(getArg(options, defaultArgs, 'nBasis))
    val ntree =  castInt(getArg(options, defaultArgs, 'ntree))
    val minNodeSize =  castInt(getArg(options, defaultArgs, 'minNodeSize))
    val mtry =  castInt(getArg(options, defaultArgs, 'mtry) match {
      case -1 => ceil((nActive + nInactive) / 3).toInt
      case x => x
    })
    val runGlobalRF = castInt(getArg(options, defaultArgs, 'runGlobalRF)) == 1
    val sampleWithReplacement = castInt(getArg(options, defaultArgs, 'sampleWithReplacement)) == 1
    val globalMinNodeSize = 10;
    val oracle = IndexedSeq.range(0,nActive)
    val runOracle = castInt(getArg(options, defaultArgs, 'runOracle)) == 1
    val threshold1 = castDouble(getArg(options, defaultArgs, 'threshold1))
    val threshold2 = castDouble(getArg(options, defaultArgs, 'threshold2))
    val nValid = castInt(getArg(options, defaultArgs, 'nValid))


    //System.out.println("Generating the data")
    // Generate the random seeds for simulating data
    val seeds = Array.fill(nPartitions)(scala.util.Random.nextLong())

    // Parallelize the seeds
    val seedsRDD = sc.parallelize(seeds, nPartitions)
    var d = -1;
    val rfparams = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(mtry, minNodeSize))
    // Generate the data
    val (dataGenerator, forestParameters) = simulationName match {
      case "Friedman1" => {
        d = 5 + nInactive
        (Friedman1Generator(nInactive), rfparams)
      }
      case "GaussianProcess" => {
        d = nActive + nInactive
        val function = FourierBasisGaussianProcessFunction.getFunction(
          nActive, nBasis, 0.05, new scala.util.Random(2015L))
        val generator = new FourierBasisGaussianProcessGenerator(function, nInactive)
        (generator, rfparams)
      }
      case other => {
        throw new IllegalArgumentException("Unknown simulation name: " + simulationName)
      }
    }

    val trainingDataRDD = seedsRDD.flatMap(
      seed => dataGenerator.generateData(
        nPointsPerPartition, 3.0, new scala.util.Random(seed)))

    //TODO:   import org.apache.spark.mllib.feature.StandardScaler scale columns
    //normalize the labels
    trainingDataRDD.persist()
    val nTrain = trainingDataRDD.count()
    val trMean = trainingDataRDD.map(point => point.label).sum/nTrain
    val trVariance = trainingDataRDD.map(point => Math.pow(point.label - trMean, 2)).sum / nTrain
    val trStdev = Math.sqrt(trVariance)

    val stdTrainingDataRDD = trainingDataRDD.map(point => new LabeledPoint((point.label-trMean)/trStdev, point.features))
    trainingDataRDD.unpersist()
    val forceTrainingData = stdTrainingDataRDD.count()
    val trainingMean = stdTrainingDataRDD.map(x => x.label).sum/forceTrainingData
    stdTrainingDataRDD.persist()


     //Train the forests
    //System.out.println("Training forests")
    val forests = DistributedForest.train(stdTrainingDataRDD, forestParameters)


    // Persist the forests in memory
    forests.persist()
    val rfTrainStart = System.currentTimeMillis
    val forceRFTraining = forests.count
    val rfTrainTime = (System.currentTimeMillis - rfTrainStart) * 1e-3

    val fit = DistributedForest.getFeatureImportance(forests)

    // Get the predictions
    //System.out.println("Training local models and getting predictions")
    val (testPredictors, testLabels) = dataGenerator.
      generateData(nTest, 0.0, scala.util.Random).
      map(point => (point.features, (point.label-trMean)/trStdev)).unzip
    val (validPredictors, validLabels) = dataGenerator.
      generateData(nValid, 0.0, scala.util.Random).
      map(point => (point.features, (point.label-trMean)/trStdev)).unzip
    //  map(point => (point.features, point.label)).unzip

    //normalize the labels
    val predictionsMean = testPredictors.map(x => trainingMean)
    val testLocRegStart = System.currentTimeMillis
    val predictionsLocalRegression = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize)
    val siloTestTime = (System.currentTimeMillis - testLocRegStart) * 1e-3

    //TODO: (Chris) get active set from the covariances of the full PNNs instead of dropping test points down forest 2x
    /**
    val activeSetStart = System.currentTimeMillis
    val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize, fit.getActiveSet(threshold1, threshold2))
    val featImpSiloTestTime = (System.currentTimeMillis - activeSetStart) * 1e-3
  */
    def validateActiveSet() = {
      var bestRMSE = -1.0
      var bestActive = -1
      var bestCorr = -1.0
      var bestPredictions = IndexedSeq(0.0)
      for(nActive <- 1 to d){
        val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatch(
          validPredictors, forests, nPnnsPerPartition, batchSize, fit.getTopFeatures(nActive))
        val currRMSE = EvaluationMetrics.rmse(predictionsActiveSet, validLabels)
        val currCorr = EvaluationMetrics.correlation(predictionsActiveSet, validLabels)
        if(bestRMSE < 0 || currRMSE < bestRMSE){
          bestRMSE = currRMSE
          bestActive = nActive
          bestCorr = currCorr
          bestPredictions = predictionsActiveSet
        }
      }
      (bestRMSE, bestCorr, bestActive, bestPredictions)
    }

    val activeSetStart = System.currentTimeMillis
    val (bestRMSE, bestCorr, bestActive, bestPredictions) = validateActiveSet()
    val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize, fit.getTopFeatures(bestActive))
    val featImpSiloTestTime = (System.currentTimeMillis - activeSetStart) * 1e-3

    val testNaiveStart = System.currentTimeMillis
    val predictionsNaiveAveraging = DistributedForest.
      predictWithNaiveAverageBatch(testPredictors, forests, batchSize)
    val naiveTestTime = (System.currentTimeMillis - testNaiveStart) * 1e-3



    var globalTrainTime = -1.0
    var globalTestTime = -1.0
    var globalRMSE = -1.0
    var globalCorr = -1.0
    var globalOracleRMSE = -1.0
    var siloOracle1RMSE = -1.0
    var siloOracle2RMSE = -1.0

    //Dont runGlobalRF with with huge datasets
    if(runOracle){
      val predictionsSiloOracle2 = DistributedForest.predictWithLocalRegressionBatch(
        testPredictors, forests, nPnnsPerPartition, batchSize, oracle)
      forests.unpersist()
      val oracleData = stdTrainingDataRDD.map(x => FeatureImportance.getActiveFeatures(x, oracle))
      stdTrainingDataRDD.unpersist()
      oracleData.persist()
      val forestParametersOracle = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(Math.ceil(oracle.length/3).toInt, minNodeSize))
      val forestsOracle1 = DistributedForest.train(oracleData, forestParametersOracle)
      val rfparamsGlobalOracle = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(Math.ceil(oracle.length/3).toInt, globalMinNodeSize))
      if(runGlobalRF){
        val globalRFOracle = RandomForest.train(oracleData.collect.toIndexedSeq, rfparamsGlobalOracle)
        val predictionsGlobalRFOracle = testPredictors.map(x => FeatureImportance.getActiveFeatures(x, oracle)).map(x => globalRFOracle.predict(x))
        globalOracleRMSE = EvaluationMetrics.rmse(predictionsGlobalRFOracle, testLabels)
      }
      oracleData.unpersist()
      val predictionsSiloOracle1 = DistributedForest.predictWithLocalRegressionBatch(
        testPredictors.map(x => FeatureImportance.getActiveFeatures(x, oracle)), forestsOracle1, nPnnsPerPartition, batchSize, oracle)
      siloOracle1RMSE = EvaluationMetrics.rmse(predictionsSiloOracle1, testLabels)
      siloOracle2RMSE = EvaluationMetrics.rmse(predictionsSiloOracle2, testLabels)
    }
    if(runGlobalRF){
      val trainingData = stdTrainingDataRDD.collect.toIndexedSeq
      val globalTrainStart = System.currentTimeMillis
      val rfparamsGlobal = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(mtry, globalMinNodeSize))
      val globalRF = RandomForest.train(trainingData, rfparamsGlobal)
      globalTrainTime = (System.currentTimeMillis - globalTrainStart) * 1e-3

      val globalTestStart = System.currentTimeMillis
      val predictionsGlobalRF = testPredictors.map(x => globalRF.predict(x))
      globalTestTime = (System.currentTimeMillis - globalTestStart) * 1e-3
      globalRMSE = EvaluationMetrics.rmse(predictionsGlobalRF, testLabels)
      globalCorr = EvaluationMetrics.correlation(predictionsGlobalRF, testLabels)

    }



    def printMetrics(predictions: IndexedSeq[Double]): Unit = {
      System.out.println("RMSE is " + EvaluationMetrics.rmse(predictions, testLabels))
      System.out.println("Correlation is " +
        EvaluationMetrics.correlation(predictions, testLabels))
    }

    val meanRMSE = EvaluationMetrics.rmse(predictionsMean, testLabels)
    val siloRMSE = EvaluationMetrics.rmse(predictionsLocalRegression, testLabels)
    val siloCorr = EvaluationMetrics.correlation(predictionsLocalRegression, testLabels)
    val featImpSiloRMSE = EvaluationMetrics.rmse(predictionsActiveSet, testLabels)
    val featImpSiloCorr = EvaluationMetrics.correlation(predictionsActiveSet, testLabels)
    val naiveRMSE = EvaluationMetrics.rmse(predictionsNaiveAveraging, testLabels)
    val naiveCorr = EvaluationMetrics.correlation(predictionsNaiveAveraging, testLabels)

    def printStuff = {
      println(simulationName + "," + nPartitions + "," + nPointsPerPartition + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","+ minNodeSize + ",")
      println (options)
      System.out.println("Dataset: " + simulationName)
      System.out.println("Number of partitions: " + nPartitions)
      System.out.println("Number of training points per partition: " + nPointsPerPartition)
      System.out.println("Total Training Points: " + nPointsPerPartition * nPartitions)
      System.out.println("dimensions: " + d)
      System.out.println("Number of test points: " + nTest)
      System.out.println("Batch size: " + batchSize)
      System.out.println("Feature Importance Active Set: " + fit.getActiveSet(threshold1, threshold2))
      System.out.println("Feature Importance Values: " + fit.getAvgImportance)
      System.out.println("Feature Importance Sum: " + fit.getSplitGains)
      System.out.println("Feature Importance Num: " + fit.getNumberOfSplits)
      // Evaluate the predictions
      System.out.println("Performance using local regression model:")
      printMetrics(predictionsLocalRegression)
      System.out.println("Performance using feature Selection")
      printMetrics(predictionsActiveSet)

      System.out.println("Performance using naive averaging")
      printMetrics(predictionsNaiveAveraging)


      System.out.println("Train time: " + rfTrainTime )
      System.out.println("Test time, local regression: " + siloTestTime)
      System.out.println("Test time, active set local regression: " + featImpSiloTestTime)
      System.out.println("Test time, naive averaging: " + naiveTestTime)
    }
    def printFeat = {
      System.out.println("Feature Importance Thresholds: " + threshold1 + " " + threshold2)
      System.out.println("Feature Importance Values: " + fit.getAvgImportance)
      System.out.println("Feature Importance Sum: " + fit.getSplitGains)
      System.out.println("Feature Importance Num: " + fit.getNumberOfSplits)
      System.out.println("Feature Importance Active Set: " + fit.getActiveSet(threshold1, threshold2))
      System.out.println("Top Feature Active Set Number: " + bestActive)
      System.out.println("Top Feature Active Set: " + fit.getTopFeatures(bestActive))
    }
    def printFormat = {
      println("siloRMSE,featImpSiloRMSE,naiveRMSE,siloCorr,featImpSiloCorr,naiveCorr,siloTestTime,featImpSiloTestTime,naiveTestTime,simulationName,nPartitions,nPointsPerPartition,nPnnsPerPartition,nTest,batchSize,nActive,nInactive," +
        "nBasis,mtry,ntree,minNodeSize")
    }
    def printFormatted = {
      println(siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","+ globalRMSE + ","
        + siloCorr + "," + featImpSiloCorr + "," + naiveCorr + ","+ globalCorr + ","
        + siloTestTime + "," + featImpSiloTestTime + "," + naiveTestTime+ "," + rfTrainTime + "," + globalTrainTime + "," + globalTestTime + ","
        + simulationName + "," + nPartitions + "," + nPointsPerPartition + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","
        + minNodeSize + "," + globalMinNodeSize + "," + fit.getActiveSet().length)
    }
    def printImportant = {
      println(
      "Simulation Name: "+simulationName +
        " nPartitions: "+nPartitions +
        " nPointsPerPartition: "+nPointsPerPartition +
        " nTest: "+nTest +
        " nActive: "+nActive +
        " nInactive: "+nInactive
      )
    }
    def printRMSEs = {
      println("RMSE,"+siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","+ globalRMSE + ","+ globalOracleRMSE + ","+ siloOracle1RMSE + ","+ siloOracle2RMSE+ ","+ meanRMSE)
    }
    //printStuff
    //printFormat
    //printFeat
    printImportant
    printFormatted
    printRMSEs


    sc.cancelAllJobs()
    sc.stop()
  }

  def createLabeledPoint(line: String) : LabeledPoint = {
    val tokens = line.split(",").map(_.toDouble)
    return new LabeledPoint(tokens.last, Vectors.dense(tokens.dropRight(1)))
  }
}
