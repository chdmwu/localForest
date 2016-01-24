package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{RandomForestParameters, TreeParameters}
import edu.berkeley.statistics.Simulations.DataGenerators.{FourierBasisGaussianProcessFunction, FourierBasisGaussianProcessGenerator, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import org.apache.spark.{SparkConf, SparkContext}

import scala.math.floor

object ExecuteDistributedSimulation {
  val usage =
    """ExecuteDistributedSimulation --simulationName [name] --nPartitions [val] --nPointsPerPartition [val] --nPnnsPerPartition [val]
      |--nTest [val] --batchSize [val] --nActive [val] --nInactive [val] --nBasis [val] --ntree [val --mtry [val] --minNodeSize [val] --runGlobalRF [val]
    """.stripMargin
  def main(args: Array[String]) = {



    val conf = new SparkConf().setAppName("DistributedForest")
    val sc = new SparkContext(conf)


    // For large simulations, may have to edit /usr/local/etc/sbtopts on MacOS if using sbt
    // or add --executor-memory XXG commandline option if using spark-submit
    if (args.length == 0) println(usage)
    val arglist = args.toList
    type OptionMap = Map[Symbol, Any]

    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      list match {
        case Nil => map
        case "--simulationName" :: value :: tail =>
          nextOption(map ++ Map('nPartitions -> value.toString), tail)
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
      case _ => System.err.println("invalid parameter: " + value)
        sys.exit(1)
    }
    val defaultArgs = Map('simulationName -> "Friedman1", 'nPartitions ->4, 'nPointsPerPartition ->2500, 'nPnnsPerPartition ->1000,
      'nTest ->1000,'batchSize -> 100,'nActive ->5,'nInactive ->0, 'nBasis -> 2500, 'ntree -> 100, 'minNodeSize -> 10).withDefaultValue(-1);
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
    //println (options)
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
      case -1 => floor((nActive + nInactive) / 3).toInt
      case x => x
    })
    val runGlobalRF = castInt(getArg(options, defaultArgs, 'runGlobalRF)) == 1



    //System.out.println("Generating the data")
    // Generate the random seeds for simulating data
    val seeds = Array.fill(nPartitions)(scala.util.Random.nextLong())

    // Parallelize the seeds
    val seedsRDD = sc.parallelize(seeds, nPartitions)
    var d = -1;
    // Generate the data
    val (dataGenerator, forestParameters) = simulationName match {
      case "Friedman1" => {
        d = 5 + nInactive
        (Friedman1Generator(nInactive),
          RandomForestParameters(ntree, true,
            TreeParameters(mtry, minNodeSize)))
      }
      case "GaussianProcess" => {
        d = nActive + nInactive
        val function = FourierBasisGaussianProcessFunction.getFunction(
          nActive, nBasis, 0.05, new scala.util.Random(2015L))
        val generator = new FourierBasisGaussianProcessGenerator(function, nInactive)
        (generator, RandomForestParameters(
          ntree, true, TreeParameters(mtry, minNodeSize)))
      }
      case other => {
        throw new IllegalArgumentException("Unknown simulation name: " + simulationName)
      }
    }

    val trainingDataRDD = seedsRDD.flatMap(
      seed => dataGenerator.generateData(
        nPointsPerPartition, 3.0, new scala.util.Random(seed)))

    trainingDataRDD.persist()
    val forceTrainingData = trainingDataRDD.count()

     //Train the forests
    //System.out.println("Training forests")
    val forests = DistributedForest.train(trainingDataRDD, forestParameters)


    // Persist the forests in memory
    forests.persist()
    val rfTrainStart = System.currentTimeMillis
    val forceRFTraining = forests.count
    val rfTrainTime = System.currentTimeMillis - rfTrainStart

    val fit = DistributedForest.getFeatureImportance(forests)

    // Get the predictions
    System.out.println("Training local models and getting predictions")
    val (testPredictors, testLabels) = dataGenerator.
      generateData(nTest, 0.0, scala.util.Random).
      map(point => (point.features, point.label)).unzip

    val testLocRegStart = System.currentTimeMillis
    val predictionsLocalRegression = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize)
    val siloTestTime = (System.currentTimeMillis - testLocRegStart) * 1e-3

    //TODO: (Chris) get active set from the covariances of the full PNNs instead of dropping test points down forest 2x
    val activeSetStart = System.currentTimeMillis
    val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize, fit.getActiveSet)
    val featImpSiloTestTime = (System.currentTimeMillis - activeSetStart) * 1e-3

    val testNaiveStart = System.currentTimeMillis
    val predictionsNaiveAveraging = DistributedForest.
      predictWithNaiveAverageBatch(testPredictors, forests, batchSize)
    val naiveTestTime = (System.currentTimeMillis - testNaiveStart) * 1e-3



    def printMetrics(predictions: IndexedSeq[Double]): Unit = {
      System.out.println("RMSE is " + EvaluationMetrics.rmse(predictions, testLabels))
      System.out.println("Correlation is " +
        EvaluationMetrics.correlation(predictions, testLabels))
    }
    val siloRMSE = EvaluationMetrics.rmse(predictionsLocalRegression, testLabels)
    val siloCorr = EvaluationMetrics.correlation(predictionsLocalRegression, testLabels)
    val featImpSiloRMSE = EvaluationMetrics.rmse(predictionsActiveSet, testLabels)
    val featImpSiloCorr = EvaluationMetrics.correlation(predictionsActiveSet, testLabels)
    val naiveRMSE = EvaluationMetrics.rmse(predictionsNaiveAveraging, testLabels)
    val naiveCorr = EvaluationMetrics.correlation(predictionsNaiveAveraging, testLabels)

    def printStuff = {
      println(simulationName + "," + nPartitions + "," + nPointsPerPartition + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","+ minNodeSize + ",")
      System.out.println("Dataset: " + simulationName)
      System.out.println("Number of partitions: " + nPartitions)
      System.out.println("Number of training points per partition: " + nPointsPerPartition)
      System.out.println("Total Training Points: " + nPointsPerPartition * nPartitions)
      System.out.println("dimensions: " + d)
      System.out.println("Number of test points: " + nTest)
      System.out.println("Batch size: " + batchSize)
      System.out.println("Feature Importance Active Set: " + fit.getActiveSet)
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
    def printFormat = {
      println("siloRMSE,featImpSiloRMSE,naiveRMSE,siloCorr,featImpSiloCorr,naiveCorr,siloTestTime,featImpSiloTestTime,naiveTestTime,simulationName,nPartitions,nPointsPerPartition,nPnnsPerPartition,nTest,batchSize,nActive,nInactive," +
        "nBasis,mtry,ntree,minNodeSize")
    }
    def printFormatted = {
      println(siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","
        + siloCorr + "," + featImpSiloCorr + "," + naiveCorr + ","
        + siloTestTime + "," + featImpSiloTestTime + "," + naiveTestTime+ ","
        + simulationName + "," + nPartitions + "," + nPointsPerPartition + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","
        + minNodeSize)
    }
    //printStuff
    //printFormat
    printFormatted


    sc.cancelAllJobs()
    sc.stop()
  }
}
