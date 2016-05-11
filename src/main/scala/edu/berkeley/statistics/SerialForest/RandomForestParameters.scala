package edu.berkeley.statistics.SerialForest

case class TreeParameters(mtry: Int, nodeSize: Long) {}

case class RandomForestParameters(ntree: Int, replace: Boolean,
                                  treeParams: TreeParameters) {}