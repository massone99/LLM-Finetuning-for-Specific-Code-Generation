import akka.actor.{Actor, ActorRef, ActorSystem, Props}

// Message requesting a coordinated task
case class CoordinatedTask(data: String)
// Message representing partial result from a worker
case class PartialResult(text: String)
// Message with final aggregated result
case class FinalResult(result: String)

// Worker that processes part of the data
class WorkerActor(workerId: Int) extends Actor {
  def receive: Receive = {
    case CoordinatedTask(data) =>
      val transformed = s"Worker-$workerId processed: $data"
      println(transformed)
      sender() ! PartialResult(transformed)
    case _ => println(s"Worker-$workerId: Unknown message.")
  }
}

// Manager that coordinates two workers and aggregates their results
class ManagerActor(worker1: ActorRef, worker2: ActorRef) extends Actor {
  private var resultCount = 0
  private var partials: List[String] = Nil
  private var originalSender: Option[ActorRef] = None

  def receive: Receive = {
    case CoordinatedTask(data) =>
      // Reset state for new request
      resultCount = 0
      partials = Nil
      originalSender = Some(sender())
      // Send the same data to two workers
      worker1 ! CoordinatedTask(data)
      worker2 ! CoordinatedTask(data)

    case PartialResult(text) =>
      // Collect partial results
      partials = text :: partials
      resultCount += 1
      // If both workers have replied, aggregate
      if (resultCount == 2) {
        val aggregated = partials.reverse.mkString(" | ")
        println(s"ManagerActor: Aggregated -> $aggregated")
        originalSender.foreach(_ ! FinalResult(aggregated))
      }

    case _ => println("ManagerActor: Unknown message.")
  }
}

object ManagerApp extends App {
  val system = ActorSystem("ManagerSystem")

  val w1 = system.actorOf(Props(new WorkerActor(1)), "worker1")
  val w2 = system.actorOf(Props(new WorkerActor(2)), "worker2")
  val manager = system.actorOf(Props(new ManagerActor(w1, w2)), "manager")

  // Send a coordinated task
  manager ! CoordinatedTask("some data")

  Thread.sleep(1000)
  system.terminate()
}