package LocalForest

import SerialForest.RandomForest
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector


/**
 * Creates a distributed localforest.
 *
 * Reads headerless csvs for training and testing data, assumes the last column is the response variable.
 */
object DistributedForest {
  val nPartitions = 5
  val numIterations = 100
  def main(args: Array[String]) = {
    var trainFile = "/home/christopher/train.csv"
    var testFile = "/home/christopher/test.csv"
    if(args.length==2) {
      val trainFile = args(0)
      val testFile = args(1)
      println("using training file: " + trainFile)
      println("using testing file: " + testFile)
    } else {
      println("Using default filenames, please input filename as commandline argument")
    }
    val conf = new SparkConf().setAppName("DistributedForest")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)
    val trainData = sc.textFile(trainFile,nPartitions).map(createLabeledPoint)
    //printList(trainData.take(5))
    //val randomForests = train(trainData)
  }
  def createLabeledPoint(line: String) : LabeledPoint = {
    val tokens = line.split(",").map(_.toDouble)
    return new LabeledPoint(tokens.last, Vectors.dense(tokens.dropRight(1)))
  }

  def train(trainData: RDD[LabeledPoint]) : RDD[RandomForest]= {
    trainData.mapPartitions[RandomForest](RandomForest.train)
  }
  def test(testData: Array[Vector], forests: RDD[RandomForest]): Array[Double] = {
    testData.map {item =>
      val pnns = forests.flatMap(_.getPNNs(item))
      val weights = forests.flatMap(_.getWeights(item))
      val model = LinearRegressionWithSGD.train(pnns, numIterations) //TODO: add weights for linear regression
      model.predict(item)
    }
  }

}
