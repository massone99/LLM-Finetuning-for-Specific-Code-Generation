import akka.actor.{Actor, ActorRef, ActorSystem, Props}

// Define messages
case class DataPoint(value: Double)
case object ComputeAverage
case class AverageResult(average: Double)

// Define the DataAggregatorActor
class DataAggregatorActor extends Actor {
  var dataPoints: List[Double] = List.empty

  def receive: Receive = {
    case DataPoint(value) =>
      dataPoints = value :: dataPoints
      println(s"DataAggregatorActor: Received data point $value")
    case ComputeAverage =>
      if (dataPoints.nonEmpty) {
        val average = dataPoints.sum / dataPoints.size
        println(f"DataAggregatorActor: Computed average = $$${average}%.2f")
        sender() ! AverageResult(average)
      } else {
        println("DataAggregatorActor: No data points to compute average.")
        sender() ! AverageResult(0.0)
      }
    case _ =>
      println("DataAggregatorActor received unknown message.")
  }
}

// Define the DataSourceActor
class DataSourceActor(aggregator: ActorRef) extends Actor {
  def receive: Receive = {
    case value: Double =>
      println(s"DataSourceActor sending data point: $value")
      aggregator ! DataPoint(value)
    case _ =>
      println("DataSourceActor received unknown message.")
  }
}

// Define the ResponderActor to handle average results
class ResponderActor extends Actor {
  def receive: Receive = {
    case AverageResult(avg) =>
      println(f"ResponderActor: The average of collected data points is $$${avg}%.2f")
    case _ =>
      println("ResponderActor received unknown message.")
  }
}

// Usage Example (for testing purposes)
object DataAggregatorApp extends App {
  val system = ActorSystem("DataAggregatorSystem")

  val aggregator = system.actorOf(Props(new DataAggregatorActor()), "aggregatorActor")
  val responder = system.actorOf(Props(new ResponderActor()), "responderActor")

  // Create data source actors
  val source1 = system.actorOf(Props(new DataSourceActor(aggregator)), "source1")
  val source2 = system.actorOf(Props(new DataSourceActor(aggregator)), "source2")
  val source3 = system.actorOf(Props(new DataSourceActor(aggregator)), "source3")

  // Send data points from sources
  source1 ! 10.5
  source2 ! 20.0
  source3 ! 30.5
  source1 ! 25.0
  source2 ! 15.5

  // Compute average
  aggregator.tell(ComputeAverage, responder)

  // Allow some time for processing before shutdown
  Thread.sleep(1000)
  system.terminate()
}