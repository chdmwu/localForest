package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{FeatureImportance, RandomForestParameters, TreeParameters, RandomForest}
import edu.berkeley.statistics.Simulations.DataGenerators.{DataReader, FourierBasisGaussianProcessFunction, FourierBasisGaussianProcessGenerator, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LassoWithSGD, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row}
import scala.math.ceil

object ExecuteDistributedSimulation {
  val usage =
    """ExecuteDistributedSimulation --simulationName [name] --nPartitions [val] --nTrain [val] --nPnnsPerPartition [val]
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
        case "--nTrain" :: value :: tail =>
          nextOption(map ++ Map('nTrain -> value.toInt), tail)
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
        case "--globalMinNodeSize" :: value :: tail =>
          nextOption(map ++ Map('globalMinNodeSize -> value.toInt), tail)
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
        case "--normalizeLabel" :: value :: tail =>
          nextOption(map ++ Map('normalizeLabel -> value.toInt), tail)
        case "--runLinear" :: value :: tail =>
          nextOption(map ++ Map('runLinear -> value.toInt), tail)

        case option :: tail => System.err.println("Unknown option "+option)
          System.err.println(usage)
          sys.exit(1)
      }
    }
    def getArg(args: Map[Symbol, Any], defaultArgs: Map[Symbol, Any], name: Symbol) : Any = args(name) match {
      case -1 => defaultArgs(name)
      case _ => args(name)
    }

    val defaultArgs = Map('simulationName -> "Friedman1", 'nPartitions ->4, 'nTrain ->2500, 'nPnnsPerPartition ->1000,
      'nTest ->1000,'batchSize -> 100,'nActive ->5,'nInactive ->0, 'nBasis -> 2500, 'ntree -> 100, 'minNodeSize -> 10, 'globalMinNodeSize -> 10, 'sampleWithReplacement -> 1
      , 'runOracle -> -1, 'threshold1 -> .1, 'threshold2 -> .33, 'nValid -> 1000, 'normalizeLabel -> 1, 'runLinear -> -1).withDefaultValue(-1);
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
    //set parameters
    //println("Setting parameters")
    val simulationName = getArg(options, defaultArgs, 'simulationName).toString()
    val nPartitions = castInt(getArg(options, defaultArgs, 'nPartitions))
    var nTrain = castInt(getArg(options, defaultArgs, 'nTrain))
    var nValid = castInt(getArg(options, defaultArgs, 'nValid))
    var nTest = castInt(getArg(options, defaultArgs, 'nTest))
    val nPnnsPerPartition = castInt(getArg(options, defaultArgs, 'nPnnsPerPartition))
    val batchSize = castInt(getArg(options, defaultArgs, 'batchSize))
    val nActive = castInt(getArg(options, defaultArgs, 'nActive))
    val nInactive = castInt(getArg(options, defaultArgs, 'nInactive))
    val nBasis =  castInt(getArg(options, defaultArgs, 'nBasis))
    val ntree =  castInt(getArg(options, defaultArgs, 'ntree))
    val minNodeSize =  castInt(getArg(options, defaultArgs, 'minNodeSize))
    val runGlobalRF = castInt(getArg(options, defaultArgs, 'runGlobalRF)) == 1
    val sampleWithReplacement = castInt(getArg(options, defaultArgs, 'sampleWithReplacement)) == 1
    val globalMinNodeSize = castInt(getArg(options, defaultArgs, 'globalMinNodeSize))
    val runOracle = castInt(getArg(options, defaultArgs, 'runOracle)) == 1 && (simulationName == "Friedman1" || simulationName == "GaussianProcess")
    val threshold1 = castDouble(getArg(options, defaultArgs, 'threshold1))
    val threshold2 = castDouble(getArg(options, defaultArgs, 'threshold2))
    val normalizeLabel = castInt(getArg(options, defaultArgs, 'normalizeLabel)) == 1;
    val runLinear = castInt(getArg(options, defaultArgs, 'runLinear)) == 1;

    val oracle = IndexedSeq.range(0,nActive)

    // Generate the data
    //println("Generating data")
    val (trainingDataRDD, validData, testData) = DataReader.getData(sc,simulationName,nPartitions,nTrain,nValid,nTest,nActive,nInactive,nBasis,normalizeLabel)
    trainingDataRDD.persist()
    val realnPartitions = trainingDataRDD.partitions.size
    assert(realnPartitions == nPartitions)
    nTrain = trainingDataRDD.count().toInt
    nValid = validData.length
    nTest = testData.length






    val validPredictors = validData.map(x => x.features)
    val validLabels = validData.map(x => x.label)
    val testPredictors = testData.map(x => x.features)
    val testLabels = testData.map(x => x.label)
    val nFeatures = testPredictors(0).size
    val trainingMean = trainingDataRDD.map(x => x.label).sum / trainingDataRDD.count

    val mtry =  castInt(getArg(options, defaultArgs, 'mtry) match {
      case -1 => ceil(nFeatures.toDouble / 3).toInt
      case x => x
    })
    println(mtry)
    val forestParameters = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(mtry, minNodeSize))

    //Train the forests
    //println("Training Forests")
    val forests = DistributedForest.train(trainingDataRDD, forestParameters)
    forests.persist()
    val rfTrainStart = System.currentTimeMillis
    val forceRFTraining = forests.count
    val rfTrainTime = (System.currentTimeMillis - rfTrainStart) * 1e-3

    //Get the feature importances
    val fit = DistributedForest.getFeatureImportance(forests)

    //Get the predictions

    //println("Getting predictions")
    //get mean prediction
    val predictionsMean = testPredictors.map(x => trainingMean)
    //get SILO prediction
    val testLocRegStart = System.currentTimeMillis
    val predictionsLocalRegression = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize)
    val siloTestTime = (System.currentTimeMillis - testLocRegStart) * 1e-3

    //TODO: (Chris) get active set from the covariances of the full PNNs instead of dropping test points down forest 2x
    def validateActiveSet() = {
      var bestRMSE = -1.0
      var bestActive = -1
      var bestCorr = -1.0
      var bestPredictions = IndexedSeq(0.0)
      for(nActive <- 1 to nFeatures){
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
    //get active set SILO prediction
    //println("Validating")
    val activeSetStart = System.currentTimeMillis
    val (bestRMSE, bestCorr, bestActive, bestPredictions) = validateActiveSet()
    val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize, fit.getTopFeatures(bestActive))
    val featImpSiloTestTime = (System.currentTimeMillis - activeSetStart) * 1e-3
    //get naive distributed RF prediction
    val testNaiveStart = System.currentTimeMillis
    val predictionsNaiveAveraging = DistributedForest.
      predictWithNaiveAverageBatch(testPredictors, forests, batchSize)
    val naiveTestTime = (System.currentTimeMillis - testNaiveStart) * 1e-3

    //println("Running Oracles/Global RF")
    //Run global RF and oracles
    var globalTrainTime = -1.0
    var globalTestTime = -1.0
    var globalRMSE = -1.0
    var globalCorr = -1.0
    var globalOracleRMSE = -1.0
    var siloOracle1RMSE = -1.0
    var siloOracle2RMSE = -1.0

    var linearRMSE = -1.0
    var lassoRMSE = -1.0
    val numIterations = 600
    val stepSize = 0.01
    //val numIterations = 10000
    if(runLinear){

      val algorithm = new LinearRegressionWithSGD()
      algorithm.setIntercept(true)
      algorithm.optimizer
        .setNumIterations(numIterations)
        .setStepSize(stepSize)
      val linModel = algorithm.run(trainingDataRDD)
      val predictionsLinear = testPredictors.map { point =>  linModel.predict(point)}
      linearRMSE = EvaluationMetrics.rmse(predictionsLinear, testLabels)
      val lassoModel = LassoWithSGD.train(trainingDataRDD, numIterations)
      val predictionsLasso = testPredictors.map { point =>  lassoModel.predict(point)}
      lassoRMSE = EvaluationMetrics.rmse(predictionsLasso, testLabels)
    }

    if(runLinear){
      val algorithm = new LassoWithSGD()
      algorithm.setIntercept(true).setValidateData(true)
      algorithm.optimizer
        .setNumIterations(numIterations)
        .setStepSize(stepSize)

      val linModel = algorithm.run(trainingDataRDD)
      val predictionsLinear = testPredictors.map { point =>  linModel.predict(point)}
      linearRMSE = EvaluationMetrics.rmse(predictionsLinear, testLabels)
      val lassoModel = LassoWithSGD.train(trainingDataRDD, numIterations)
      val predictionsLasso = testPredictors.map { point =>  lassoModel.predict(point)}
      lassoRMSE = EvaluationMetrics.rmse(predictionsLasso, testLabels)
    }


    //Dont runGlobalRF with with huge datasets
    if(runOracle){
      //println("Running silo oracle 2")
      val predictionsSiloOracle2 = DistributedForest.predictWithLocalRegressionBatch(
        testPredictors, forests, nPnnsPerPartition, batchSize, oracle)
      forests.unpersist()
      val oracleData = trainingDataRDD.map(x => FeatureImportance.getActiveFeatures(x, oracle))
      oracleData.persist()
      val forestParametersOracle = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(Math.ceil(oracle.length/3).toInt, minNodeSize))
     // println("Running silo oracle 1")
      val forestsOracle1 = DistributedForest.train(oracleData, forestParametersOracle)
      val rfparamsGlobalOracle = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(Math.ceil(oracle.length/3).toInt, globalMinNodeSize))
      if(runGlobalRF){
        //println("Running Global oracle 1")
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
      val trainingData = trainingDataRDD.collect.toIndexedSeq
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
      println(simulationName + "," + nPartitions + "," + nTrain + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","+ minNodeSize + ",")
      println (options)
      System.out.println("Dataset: " + simulationName)
      System.out.println("Number of partitions: " + nPartitions)
      System.out.println("Number of training points per partition: " + nTrain)
      System.out.println("Total Training Points: " + nTrain * nPartitions)
      //System.out.println("dimensions: " + d)
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
      println("siloRMSE,featImpSiloRMSE,naiveRMSE,siloCorr,featImpSiloCorr,naiveCorr,siloTestTime,featImpSiloTestTime,naiveTestTime,simulationName,nPartitions,nTrain,nPnnsPerPartition,nTest,batchSize,nActive,nInactive," +
        "nBasis,mtry,ntree,minNodeSize")
    }
    def printFormatted = {
      println(siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","+ globalRMSE + ","
        + siloCorr + "," + featImpSiloCorr + "," + naiveCorr + ","+ globalCorr + ","
        + siloTestTime + "," + featImpSiloTestTime + "," + naiveTestTime+ "," + rfTrainTime + "," + globalTrainTime + "," + globalTestTime + ","
        + simulationName + "," + nPartitions + "," + nTrain + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","
        + minNodeSize + "," + globalMinNodeSize + "," + bestActive)
    }
    def printImportant = {
      println(
      "Simulation Name: "+simulationName +
        " nPartitions: "+nPartitions +
        " nTrain: "+nTrain +
        " nValid: "+nValid +
        " nTest: "+nTest +
        " nFeatures: " + nFeatures +
        " nActive: "+nActive +
        " nInactive: "+nInactive +
        " Total Features: " + nFeatures +
        " Number of Features Selected: " + bestActive
      )
      System.out.println("Top Feature Active Set: " + fit.getTopFeatures(bestActive))
    }
    def printRMSEs = {
      println("SiloRMSE,FeatImpRMSE,NaiveRMSE,GlobalRMSE,GlobalOracle1RMSE,SiloOracle1RMSE,SiloOracle2RMSE,MeanRMSE,linearRMSE,lassoRMSE")
      println(siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","+ globalRMSE + ","+ globalOracleRMSE + ","+ siloOracle1RMSE + ","+ siloOracle2RMSE+ ","+ meanRMSE+ ","+ linearRMSE+ ","+ lassoRMSE)
    }
    def printRunTimes = {
      println("rfTrainTime,globalTrainTime,siloTestTime,featImpSiloTestTime,naiveTestTime,globalTestTime")
      println(rfTrainTime + "," + globalTrainTime + "," + siloTestTime + ","+ featImpSiloTestTime + ","+ naiveTestTime + ","+ globalTestTime)

    }
    //printStuff
    //printFormat
    //printFeat
    printImportant
   // printFormatted
    printRunTimes
    printRMSEs



    sc.cancelAllJobs()
    sc.stop()
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
}
