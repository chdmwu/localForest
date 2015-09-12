package SerialForest

import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

class RandomForest (trees: Seq[Tree], trainingData: IndexedSeq[LabeledPoint]) extends Serializable {

  private def getTopPNNIndicesAndWeights(testPoint: mllibVector, numPNNs: Int): IndexedSeq[(Int, Double)] = {
    val indexCounts = scala.collection.mutable.OpenHashMap.empty[Int, Double]

    trees.map(tree => {
      val indices = tree.getPNNIndices(testPoint)
      indices.foreach(idx => indexCounts.update(idx, indexCounts.get(idx).getOrElse(0.0) + 1.0))
    })

    val totalWeight = indexCounts.values.sum

    if (numPNNs >= indexCounts.size) {
      indexCounts.mapValues(_ / totalWeight).toIndexedSeq
    } else {
      val weightCutoff = QuickSelect(indexCounts.values.toArray, numPNNs, scala.util.Random)
      indexCounts.filter(_._2 >= weightCutoff).mapValues(_ / totalWeight).toIndexedSeq
    }
  }

  def getTopPNNsAndWeights(testPoint: mllibVector, numPNNs: Int): IndexedSeq[(LabeledPoint, Double)] = {
    this.getTopPNNIndicesAndWeights(testPoint, numPNNs).map{
      case (index: Int, weight: Double) => (trainingData(index), weight)}
  }

  def predict(testPoint: mllibVector): Double = {
    trees.map(_.predict(testPoint)).sum / trees.length
  }
}

object RandomForest {
  private def getSamplingFn(parameters: RandomForestParameters):
  (Int, Int, scala.util.Random) => Array[Int] =
    if (parameters.replace) {
      RandomSampling.sampleWithReplacement
    } else {
      RandomSampling.sampleWithoutReplacement
    }

  def parTrain(trainingData: IndexedSeq[LabeledPoint],
               parameters: RandomForestParameters): RandomForest = {
    val samplingFn = getSamplingFn(parameters)
    new RandomForest(List.fill(parameters.ntree)(scala.util.Random.nextLong).par.map(
      seed => Tree.train(
        samplingFn(trainingData.length, trainingData.length,
          new scala.util.Random(seed)),
        trainingData, parameters.treeParams, seed)).seq, trainingData)
  }

  def train(trainingData: IndexedSeq[LabeledPoint],
            parameters: RandomForestParameters): RandomForest = {
    val samplingFn = getSamplingFn(parameters)
    new RandomForest(List.fill(parameters.ntree)(scala.util.Random.nextLong).map(
      seed => Tree.train(
        samplingFn(trainingData.length, trainingData.length,
          new scala.util.Random(seed)),
        trainingData, parameters.treeParams, seed)), trainingData)
  }
}