package edu.berkeley.statistics.SerialForest

/**
 * Created by Adam on 9/4/15.
 */

case class TreeParameters(mtry: Int, nodeSize: Long) {}

case class RandomForestParameters(ntree: Int, replace: Boolean,
                                  treeParams: TreeParameters) {}