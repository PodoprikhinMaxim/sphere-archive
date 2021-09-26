
import org.apache.spark.graphx.GraphLoader
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

import java.io.PrintWriter

object PageRankExample {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext
    val spark = SQLContext(sc)

    val graph = GraphLoader.edgeListFile(sc, "InOut/soc-LiveJournal1_2.txt")
    val ranks = graph.pageRank(0.0001).vertices
    val users = sc.textFile("InOut/nodes.txt").map { line =>
      val fields = line.split(",")
      (fields(0).toLong, fields(1))
    }
    val ranksByUsername = users.join(ranks).map {
      case (id, (username, rank)) => (username, rank)
    }
    new PrintWriter("InOut/filename.txt") { write(ranksByUsername.collect().mkString("\n")); close }
  }
}w