package edu.berkeley.statistics.SerialForest

import breeze.linalg.{DenseVector, DenseMatrix}

import scala.collection._
import org.apache.spark.mllib.linalg.{Vector => mllibVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
/**
 * Created by christopher on 1/10/16.
 */
class FeatureImportance (nFeatures: Int) extends Serializable {
  var numberOfSplits = mutable.ArrayBuffer.fill(nFeatures)(0.0)
  var splitGains = mutable.ArrayBuffer.fill(nFeatures)(0.0)



  def updateGain(featureIndex: Int, gain: Double): Unit ={
    numberOfSplits(featureIndex) += 1
    splitGains(featureIndex) += gain
  }

  def updateGain(splits: IndexedSeq[Double], gains: IndexedSeq[Double]): Unit ={
    this.numberOfSplits = (this.numberOfSplits, splits).zipped.map(_ + _)
    this.splitGains = (this.splitGains, gains).zipped.map(_ + _)
  }

  def getNumberOfSplits: IndexedSeq[Double] = {
    numberOfSplits.toIndexedSeq
  }

  def getSplitGains: IndexedSeq[Double] = {
    splitGains.toIndexedSeq
  }

  def getAvgImportance = {
    (splitGains, numberOfSplits).zipped.map(_ / _).toIndexedSeq
  }

  def +(that: FeatureImportance): FeatureImportance = {
    val splits = (this.getNumberOfSplits, that.getNumberOfSplits).zipped.map(_ + _)
    val gains = (this.getSplitGains, that.getSplitGains).zipped.map(_ + _)
    val fit = new FeatureImportance(this.nFeatures)
    fit.updateGain(splits, gains)
    fit
  }

  /**def getActiveSet: IndexedSeq[Int] = {
    val minRemoved = getAvgImportance.map(x => x - getAvgImportance.min)
    val maxImp = minRemoved.max
    minRemoved.map(_/maxImp).zipWithIndex.filter(_._1 >= threshold).map(_._2).toIndexedSeq
  }*/
  /**def getActiveSet(): IndexedSeq[Int] = {
    val minRemoved = getAvgImportance.map(x => x - getAvgImportance.min)
    val criteria1 = minRemoved.map(_/minRemoved.max) //.zipWithIndex.filter(_._1 >= threshold1).map(_._2).toIndexedSeq
    val criteria2 = getAvgImportance
    val criteria2Threshold = getAvgImportance.max * threshold2
    criteria1.zip(criteria2).zipWithIndex.filter(x => (x._1._1 >= threshold1 || x._1._2 >= criteria2Threshold)).map(_._2)
  }*/
  def getActiveSet(threshold1 : Double = .1, threshold2 : Double = .33): IndexedSeq[Int] = {
    val minRemoved = getSplitGains.map(x => x - getSplitGains.min)
    val criteria1 = minRemoved.map(_/minRemoved.max) //.zipWithIndex.filter(_._1 >= threshold1).map(_._2).toIndexedSeq
    val criteria2 = getSplitGains
    val criteria2Threshold = getSplitGains.max * threshold2
    criteria1.zip(criteria2).zipWithIndex.filter(x => (x._1._1 >= threshold1 || x._1._2 >= criteria2Threshold)).map(_._2)
  }

def getTopFeatures(nFeatures: Int): IndexedSeq[Int] = {
val n = nFeatures > getSplitGains.length || nFeatures < 0 match{
case true =>  getSplitGains.length
case false => nFeatures
}
getSplitGains.zipWithIndex.sortWith(_._1 > _._1).take(n).map(_._2).sorted
}

def getSortedFeatures: IndexedSeq[Int] = {
getSplitGains.zipWithIndex.sortWith(_._1 > _._1).map(_._2)
}
}

object FeatureImportance {
  def getActiveFeatures(point: LabeledPoint, activeSet: IndexedSeq[Int]) : LabeledPoint = {
    activeSet match {
      case null => point
      case _ => new LabeledPoint(point.label, Vectors.dense(activeSet.collect(point.features.toArray).toArray))
    }
  }
    def getActiveFeatures(point: mllibVector, activeSet: IndexedSeq[Int]) : mllibVector = {
      activeSet match {
        case null => point
        case _ => Vectors.dense(activeSet.collect(point.toArray).toArray)
      }
  }
}
