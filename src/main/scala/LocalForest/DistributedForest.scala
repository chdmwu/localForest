package LocalForest

import SerialForest.RandomForest
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors


object DistributedForest {
  val nPartitions = 5;
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
    printList(trainData.take(5))
    //val randomForests = train(trainData)
  }
  def createLabeledPoint(line: String) : LabeledPoint = {
    val tokens = line.split(",").map(_.toDouble)
    return new LabeledPoint(tokens.last, Vectors.dense(tokens.dropRight(1)))
  }

  def train(trainData: RDD[LabeledPoint]) : RDD[RandomForest]= {
    trainData.mapPartitions[RandomForest](RandomForest.train)
  }
  def test(testData: Array[LabeledPoint], forests: RDD[RandomForest]): Array[Double] = ???


  //Debug stuff
  def printList(args: Array[_]): Unit = {
    args.foreach(println)
  }
}
