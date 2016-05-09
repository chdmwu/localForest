package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{FeatureImportance, RandomForestParameters, TreeParameters, RandomForest}
import edu.berkeley.statistics.Simulations.DataGenerators.{DataReader, FourierBasisGaussianProcessFunction, FourierBasisGaussianProcessGenerator, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LassoWithSGD, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row}
import scala.collection.mutable.ListBuffer
import scala.math._
import org.apache.spark.mllib.tree.{RandomForest => mllibRF} //rename mllib's RandomForest

object ExecuteDistributedSimulation {
  val usage =
    """ExecuteDistributedSimulation --trainFile [name] --nPartitions [val] --nTrain [val] --nPnnsPerPartition [val]
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
        case "--trainFile" :: value :: tail =>
          nextOption(map ++ Map('trainFile -> value.toString), tail)
        case "--validFile" :: value :: tail =>
          nextOption(map ++ Map('validFile -> value.toString), tail)
        case "--testFile" :: value :: tail =>
          nextOption(map ++ Map('testFile -> value.toString), tail)
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
        case "--nValid" :: value :: tail =>
          nextOption(map ++ Map('nValid -> value.toInt), tail)
        case "--normalizeLabel" :: value :: tail =>
          nextOption(map ++ Map('normalizeLabel -> value.toInt), tail)
        case "--normalizeFeatures" :: value :: tail =>
          nextOption(map ++ Map('normalizeFeatures -> value.toInt), tail)
        case "--runLinear" :: value :: tail =>
          nextOption(map ++ Map('runLinear -> value.toInt), tail)
        case "--runMllib" :: value :: tail =>
          nextOption(map ++ Map('runMllib -> value.toInt), tail)
        case "--mllibMaxDepth" :: value :: tail =>
          nextOption(map ++ Map('mllibMaxDepth -> value.toInt), tail)
        case "--mllibBins" :: value :: tail =>
          nextOption(map ++ Map('mllibBins -> value.toInt), tail)
        case "--mllibMinNodeSize" :: value :: tail =>
          nextOption(map ++ Map('mllibMinNodeSize -> value.toInt), tail)
        case "--debug" :: value :: tail =>
          nextOption(map ++ Map('debug -> value.toInt), tail)

        case option :: tail => System.err.println("Unknown option "+option)
          System.err.println(usage)
          sys.exit(1)
      }
    }
    def getArg(args: Map[Symbol, Any], defaultArgs: Map[Symbol, Any], name: Symbol) : Any = args(name) match {
      case "not set" => defaultArgs(name)
      case _ => args(name)
    }

    val defaultArgs = Map('trainFile -> "Friedman1",'validFile -> "",'testFile -> "", 'nPartitions ->4, 'nTrain ->2500, 'nPnnsPerPartition ->1000,
      'nTest ->1000,'batchSize -> 100,'nActive ->5,'nInactive ->0, 'nBasis -> 2500, 'ntree -> 100, 'minNodeSize -> 10, 'globalMinNodeSize -> 10, 'sampleWithReplacement -> 1
      , 'runOracle -> -1, 'threshold1 -> .1, 'threshold2 -> .33, 'nValid -> 1000, 'normalizeLabel -> 1, 'normalizeFeatures -> 1, 'runLinear -> -1, 'debug -> -1,
    'runMllib -> -1, 'mllibMaxDepth -> 10,'mllibBins -> 32,'mllibMinNodeSize -> 10).withDefaultValue(-1);
    val options = nextOption(Map().withDefaultValue("not set"),arglist)

    //set parameters
    //println("Setting parameters")
    val trainFile = getArg(options, defaultArgs, 'trainFile).toString()
    val validFile = getArg(options, defaultArgs, 'validFile).toString()
    val testFile = getArg(options, defaultArgs, 'testFile).toString()
    val nPartitions = castInt(getArg(options, defaultArgs, 'nPartitions))
    var nTrain = castInt(getArg(options, defaultArgs, 'nTrain))
    var nValid = castInt(getArg(options, defaultArgs, 'nValid))
    var nTest = castInt(getArg(options, defaultArgs, 'nTest))
    val batchSize = castInt(getArg(options, defaultArgs, 'batchSize))
    val nActive = castInt(getArg(options, defaultArgs, 'nActive))
    val nInactive = castInt(getArg(options, defaultArgs, 'nInactive))
    val nBasis =  castInt(getArg(options, defaultArgs, 'nBasis))
    val ntree =  castInt(getArg(options, defaultArgs, 'ntree))
    val minNodeSize =  castInt(getArg(options, defaultArgs, 'minNodeSize))
    val runGlobalRF = castInt(getArg(options, defaultArgs, 'runGlobalRF)) == 1
    val sampleWithReplacement = castInt(getArg(options, defaultArgs, 'sampleWithReplacement)) == 1
    val globalMinNodeSize = castInt(getArg(options, defaultArgs, 'globalMinNodeSize))
    val runOracle = castInt(getArg(options, defaultArgs, 'runOracle)) == 1 //&& (trainFile == "Friedman1" || trainFile == "GaussianProcess")
    val normalizeLabel = castInt(getArg(options, defaultArgs, 'normalizeLabel)) == 1;
    val normalizeFeatures = castInt(getArg(options, defaultArgs, 'normalizeFeatures)) == 1;
    val runLinear = castInt(getArg(options, defaultArgs, 'runLinear)) == 1;
    val runMllib = castInt(getArg(options, defaultArgs, 'runMllib)) == 1;
    val mllibMaxDepth = castInt(getArg(options, defaultArgs, 'mllibMaxDepth));
    val mllibBins = castInt(getArg(options, defaultArgs, 'mllibBins));
    val mllibMinNodeSize = castInt(getArg(options, defaultArgs, 'mllibMinNodeSize));
    val debug = castInt(getArg(options, defaultArgs, 'debug)) == 1;

    def printlnd(x:Any) = {
      debug match {
        case true => println(x)
        case _ =>
      }
    }

    val oracle = IndexedSeq.range(0,nActive)

    // Generate the data
    //println("Generating data")
    val (trainingDataRDD, validData, testData) = DataReader.getData(sc,trainFile,nPartitions,nTrain,nValid,nTest,validFile,testFile,nActive,nInactive,nBasis,normalizeLabel,normalizeFeatures)




    trainingDataRDD.persist()
    val realnPartitions = trainingDataRDD.partitions.size
    assert(realnPartitions == nPartitions)
    nTrain = trainingDataRDD.count().toInt
    nValid = validData.length
    nTest = testData.length

    //trainingDataRDD.collect().map(println(_))




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
    val nPnnsPerPartition = castInt(getArg(options, defaultArgs, 'nPnnsPerPartition)) match {
      case -1 => ceil(nTrain.toDouble / 10).toInt
      case x => x
    }
    //println(mtry)
    val forestParameters = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(mtry, minNodeSize))

    //Train the forests
    printlnd("Training Forests")
    val forests = DistributedForest.train(trainingDataRDD, forestParameters)
    forests.persist()
    val rfTrainStart = System.currentTimeMillis
    val forceRFTraining = forests.count
    val rfTrainTime = (System.currentTimeMillis - rfTrainStart) * 1e-3

    //Get the feature importances
    val fit = DistributedForest.getFeatureImportance(forests)

    //Get the predictions

    printlnd("Getting predictions")
    //get mean prediction
    val predictionsMean = testPredictors.map(x => trainingMean)


    printlnd("Getting SILO predictions")
    val testLocRegStart = System.currentTimeMillis
    val predictionsLocalRegression = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize)
    val siloTestTime = (System.currentTimeMillis - testLocRegStart) * 1e-3


    def validateActiveSet() = {
      val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatchValidate(validPredictors, forests, nPnnsPerPartition, batchSize, fit)
      val rmses = predictionsActiveSet.map(predictions => EvaluationMetrics.rmse(predictions, validLabels))
      val bestActive = rmses.zip(1 to nFeatures).minBy(_._1)._2
      val bestRMSE = rmses.min
      printlnd("best rmse:")
      printlnd(bestRMSE)
      println(rmses)
      bestActive
    }
    //get active set SILO prediction
    printlnd("Validating")
    val activeSetStart = System.currentTimeMillis
    val bestActive = validateActiveSet()
    printlnd("Getting Feat Imp SILO predictions")
    val predictionsActiveSet = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize, fit.getTopFeatures(bestActive))
    val featImpSiloTestTime = (System.currentTimeMillis - activeSetStart) * 1e-3
    //get naive distributed RF prediction

    printlnd("Getting Naive predictions")
    val testNaiveStart = System.currentTimeMillis
    val predictionsNaiveAveraging = DistributedForest.
      predictWithNaiveAverageBatch(testPredictors, forests, batchSize)
    val naiveTestTime = (System.currentTimeMillis - testNaiveStart) * 1e-3

    var siloActiveSetRMSE = -1.0
    val runFeatureImportance2=true; //TODO get rid of this
    if(runFeatureImportance2){
      val featureImportanceSet = fit.getTopFeatures(bestActive)
      val trainDataActiveSet = trainingDataRDD.map(x=>FeatureImportance.getActiveFeatures(x,featureImportanceSet))
      val forestParametersFeatures = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(ceil(bestActive.toDouble / 3).toInt, minNodeSize))
      val forestsFeatures = DistributedForest.train(trainDataActiveSet, forestParametersFeatures)

      val testPredictorsActiveSet = testPredictors.map(x=>FeatureImportance.getActiveFeatures(x,featureImportanceSet))
      val predictionsActiveSet2 = DistributedForest.predictWithLocalRegressionBatch(
        testPredictorsActiveSet, forestsFeatures, nPnnsPerPartition, batchSize)
      siloActiveSetRMSE = EvaluationMetrics.rmse(predictionsActiveSet2, testLabels)
    }




    //println("Running Oracles/Global RF")
    //Run global RF and oracles
    var globalTrainTime = -1.0
    var globalTestTime = -1.0
    var globalRMSE = -1.0
    var globalCorr = -1.0
    var globalOracleRMSE = -1.0
    var siloOracle1RMSE = -1.0
    var siloOracle2RMSE = -1.0
    var mllibTrainTime = -1.0
    var mllibTestTime = -1.0
    var mllibRMSE = -1.0
    var mllibCorr = -1.0



    var linearRMSE = -1.0
    var lassoRMSE = -1.0
    val numIterations = 300
    val stepSize = 0.01
    val lambdas = List(.0001,.001,.01,.1,1.0,10.0,100.0,1000.0)

    //val numIterations = 10000
    if(runLinear){
      printlnd("Running Linear Regression")
      val linModel = LinearRegressionWithSGD.train(trainingDataRDD,numIterations,stepSize)
      val predictionsLinear = testPredictors.map { point =>  linModel.predict(point)}
      linearRMSE = EvaluationMetrics.rmse(predictionsLinear, testLabels)

      printlnd("Running Lasso Regression")
      //validation
      val validRMSEs = lambdas.map(
        lambda => {
          val lassoModel = LassoWithSGD.train(trainingDataRDD,numIterations,stepSize,lambda)
          val validPrediction =  validPredictors.map { point =>  lassoModel.predict(point)}
          val validRMSE = EvaluationMetrics.rmse(predictionsLinear, testLabels)
          validRMSE
        })
      val bestLambda = validRMSEs.zip(lambdas).minBy(_._1)._2
      val lassoModel = LassoWithSGD.train(trainingDataRDD,numIterations,stepSize,bestLambda)
      val predictionsLasso = testPredictors.map{ point =>  lassoModel.predict(point)}
      lassoRMSE = EvaluationMetrics.rmse(predictionsLasso, testLabels)
    }


    //Dont runGlobalRF with with huge datasets
    if(runOracle){
      printlnd("Running Oracle")
      printlnd(oracle)
      //println("Running silo oracle 2")
      val predictionsSiloOracle2 = DistributedForest.predictWithLocalRegressionBatch(
        testPredictors, forests, nPnnsPerPartition, batchSize, oracle)
      forests.unpersist()
      val oracleData = trainingDataRDD.map(x => FeatureImportance.getActiveFeatures(x, oracle))
      oracleData.persist()
      val forestParametersOracle = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(Math.ceil(oracle.length.toDouble/3).toInt, minNodeSize))
     // println("Running silo oracle 1")
      val forestsOracle1 = DistributedForest.train(oracleData, forestParametersOracle)
      val rfparamsGlobalOracle = RandomForestParameters(ntree, sampleWithReplacement, TreeParameters(Math.ceil(oracle.length.toDouble/3).toInt, globalMinNodeSize))
      if(runGlobalRF){
        //println("Running Global oracle 1")

        val trainingData = trainingDataRDD.collect.toIndexedSeq

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
      printlnd("Running Global RF")
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
    if(runMllib){
      printlnd("Running MLLib")
      val treeStrategy = Strategy.defaultStrategy("Regression")
      treeStrategy.setMinInstancesPerNode(mllibMinNodeSize)
      treeStrategy.setMaxDepth(mllibMaxDepth)
      treeStrategy.setMaxBins(mllibBins)
      val featureSubsetStrategy = "auto" // Let the algorithm choose.
      val rfTrainStart = System.currentTimeMillis
      val model = mllibRF.trainRegressor(trainingDataRDD,
        treeStrategy, ntree, featureSubsetStrategy, seed = 12345)
      mllibTrainTime = (System.currentTimeMillis - rfTrainStart)* 1e-3

      val mllibStart = System.currentTimeMillis
      val mllibPredictions = testPredictors.map(model.predict(_))
      mllibTestTime = System.currentTimeMillis - mllibStart
      mllibRMSE = EvaluationMetrics.rmse(mllibPredictions, testLabels)
      mllibCorr = EvaluationMetrics.correlation(mllibPredictions, testLabels)
    }


    def printMetrics(predictions: IndexedSeq[Double]): Unit = {
      System.out.println("RMSE is " + EvaluationMetrics.rmse(predictions, testLabels))
      System.out.println("Correlation is " +
        EvaluationMetrics.correlation(predictions, testLabels))
    }
    //printlnd("rfPred")
    //printlnd(predictionsLocalRegression)
    val meanRMSE = EvaluationMetrics.rmse(predictionsMean, testLabels)
    val siloRMSE = EvaluationMetrics.rmse(predictionsLocalRegression, testLabels)
    val siloCorr = EvaluationMetrics.correlation(predictionsLocalRegression, testLabels)
    val featImpSiloRMSE = EvaluationMetrics.rmse(predictionsActiveSet, testLabels)
    val featImpSiloCorr = EvaluationMetrics.correlation(predictionsActiveSet, testLabels)
    val naiveRMSE = EvaluationMetrics.rmse(predictionsNaiveAveraging, testLabels)
    val naiveCorr = EvaluationMetrics.correlation(predictionsNaiveAveraging, testLabels)

    def printStuff = {
      println(trainFile + "," + nPartitions + "," + nTrain + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","+ minNodeSize + ",")
      println (options)
      System.out.println("Dataset: " + trainFile)
      System.out.println("Number of partitions: " + nPartitions)
      System.out.println("Number of training points per partition: " + nTrain)
      System.out.println("Total Training Points: " + nTrain * nPartitions)
      //System.out.println("dimensions: " + d)
      System.out.println("Number of test points: " + nTest)
      System.out.println("Batch size: " + batchSize)
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
      System.out.println("Feature Importance Values: " + fit.getAvgImportance)
      System.out.println("Feature Importance Sum: " + fit.getSplitGains)
      System.out.println("Feature Importance Num: " + fit.getNumberOfSplits)
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
        + trainFile + "," + nPartitions + "," + nTrain + "," + nPnnsPerPartition + "," + nTest + "," + batchSize + "," + nActive + "," + nInactive + "," + nBasis + "," + mtry + ","+ ntree + ","
        + minNodeSize + "," + globalMinNodeSize + "," + bestActive)
    }
    def printImportant = {
      println(
      "trainFile Name: "+trainFile +
        " ntree: "+ntree +
        " nPartitions: "+nPartitions +
        " nPnnsPerPartition " + nPnnsPerPartition +
        " nTrain: "+nTrain +
        " nValid: "+nValid +
        " nTest: "+nTest +
        " nFeatures: " + nFeatures +
        " nActive: "+nActive +
        " nInactive: "+nInactive +
        " Total Features: " + nFeatures +
        " MinNodeSize: " + minNodeSize +
        " GlobalMinNodeSize: " + globalMinNodeSize +
        " Number of Features Selected: " + bestActive
      )
      System.out.println("Top Feature Active Set: " + fit.getTopFeatures(bestActive))
    }
    def printRMSEs = {
      println("SiloRMSE,FeatImpRMSE,NaiveRMSE,GlobalRMSE,GlobalOracle1RMSE,SiloOracle1RMSE,SiloOracle2RMSE,MeanRMSE,linearRMSE,lassoRMSE,siloActiveSetRMSE")
      println(siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","+ globalRMSE + ","+ globalOracleRMSE + ","+ siloOracle1RMSE + ","+ siloOracle2RMSE+ ","+ meanRMSE+ ","+ linearRMSE+ ","+ lassoRMSE+","+siloActiveSetRMSE)
    }
    def printRunTimes = {
      println("rfTrainTime,globalTrainTime,siloTestTime,featImpSiloTestTime,naiveTestTime,globalTestTime")
      println(rfTrainTime + "," + globalTrainTime + "," + siloTestTime + ","+ featImpSiloTestTime + ","+ naiveTestTime + ","+ globalTestTime)

    }
    def printForMatlab = {
      println("nFeatures,bestActive,SiloRMSE,FeatImpRMSE,NaiveRMSE,GlobalRMSE,MeanRMSE,LinearRMSE,LassoRMSE,mllibRMSE")
      println(nFeatures+ "," + bestActive + "," + siloRMSE + "," + featImpSiloRMSE + "," + naiveRMSE + ","+ globalRMSE + "," + meanRMSE+ ","+ linearRMSE+ ","+ lassoRMSE+ ","+ mllibRMSE)

    }
    def printMllib = {
      println("mllibMinNodeSize,mllibMaxDepth,mllibBins,mllibTrainTime,mllibTestTime")
      println(mllibMinNodeSize+ "," + mllibMaxDepth + "," + mllibBins+ "," + mllibTrainTime + "," + mllibTestTime)
    }
    //printStuff
    //printFormat
    //printFeat
    printImportant
   // printFormatted
    printRunTimes
    printRMSEs
    printForMatlab
    if(runMllib){
      printMllib
    }



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
